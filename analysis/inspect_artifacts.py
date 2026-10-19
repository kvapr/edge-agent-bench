#!/usr/bin/env python3
"""Describe the campaign's model artifacts without reading any tensor, and
write the result to analysis/data/artifacts.json for analysis/report_stack.py.

For every model in campaign/v2_models.json:

- GGUF (cpu and npu_llama paths): only the header and the key-value metadata
  section are read; the tensor-info and tensor-data sections never are. Kept:
  file name and size, architecture, file type, quantisation version, whether
  an importance matrix was used, the f16 KV-cache bytes per token of context
  (from the attention geometry; not modelled for Gemma 4, whose sliding-window
  and shared-KV layers need tensor shapes), and the embedded chat template.
- Compiled bundle (tier A): the zip listing and its small text members
  (genie_config.json, metadata.json, tokenizer_config.json; never a .bin):
  zip size, context size, context-binary parts and the largest of them, and
  the two prompt constructions a bundle carries, a Jinja chat template and a
  prefix/suffix table.

A template "references request-level tools" when a Jinja statement or
expression uses the top-level variable `tools`; string literals and lookups
such as message['tools'] do not count. Up to three template lines are kept as
evidence.

Artifact hashes are not recomputed (that would read the tensors). The sha256
in the results' server records is copied, and compared with the hash in the
artifacts directory's download.log where that log lists the file.

Usage:
  python3 analysis/inspect_artifacts.py ARTIFACTS_DIR [--results results/v2] [--out analysis/data/artifacts.json]
"""
import argparse
import hashlib
import json
import os
import re
import struct
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))

# GGUF value types (gguf.h)
T_STR, T_ARR = 8, 9
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
FILE_TYPES = {0: "ALL_F32", 1: "MOSTLY_F16", 2: "MOSTLY_Q4_0", 3: "MOSTLY_Q4_1", 7: "MOSTLY_Q8_0"}
MAX_ARRAY = 64  # arrays longer than this (token lists, merges) are skipped, not read
EXCERPT_LINES = 3
EXCERPT_CHARS = 200

_BLOCK = re.compile(r"\{[%{](.*?)[%}]\}", re.S)
_STRING = re.compile(r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'", re.S)
_TOOLS_VAR = re.compile(r"(?<![\w.\[])tools(?!\w)")
_TOOLS_FIELD = re.compile(r"message\[['\"]tools['\"]\]|message\.tools\b")


# ------------------------------------------------------------------ GGUF
class _Reader:
    def __init__(self, f):
        self.f = f

    def take(self, fmt):
        n = struct.calcsize(fmt)
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError
        return struct.unpack(fmt, b)[0]

    def string(self, keep=True):
        n = self.take("<Q")
        if keep:
            return self.f.read(n).decode("utf-8", "replace")
        self.f.seek(n, 1)
        return None

    def value(self, t):
        if t in SCALAR:
            return self.take(SCALAR[t])
        if t == T_STR:
            return self.string()
        if t != T_ARR:
            raise ValueError(f"unknown GGUF value type {t}")
        et, n = self.take("<I"), self.take("<Q")
        if n <= MAX_ARRAY:
            return [self.value(et) for _ in range(n)]
        if et in SCALAR:
            self.f.seek(n * struct.calcsize(SCALAR[et]), 1)
        elif et == T_STR:
            for _ in range(n):
                self.string(keep=False)
        else:
            for _ in range(n):
                self.value(et)
        return {"array_len": n}


def gguf_metadata(path):
    """The key-value section of a GGUF file; stops before the tensor info."""
    with open(path, "rb", buffering=1 << 20) as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        r = _Reader(f)
        version, n_tensors, n_kv = r.take("<I"), r.take("<Q"), r.take("<Q")
        kv = {}
        for _ in range(n_kv):
            k = r.string()
            kv[k] = r.value(r.take("<I"))
    return version, n_tensors, kv


def kv_bytes_per_token(kv):
    """f16 K plus V bytes per token of context over the attention layers."""
    arch = kv["general.architecture"]
    if arch == "gemma4":
        return None

    def g(k, default=None):
        return kv.get(f"{arch}.{k}", default)
    head = g("embedding_length") // g("attention.head_count")
    kl, vl = g("attention.key_length", head), g("attention.value_length", head)
    hkv = g("attention.head_count_kv")
    if isinstance(hkv, list):              # hybrid models list it per layer; 0 marks a non-attention layer
        return sum(h * (kl + vl) for h in hkv) * 2
    layers = g("block_count")
    if g("full_attention_interval"):       # Qwen3.5: every n-th layer is attention, the rest linear
        layers //= g("full_attention_interval")
    return layers * hkv * (kl + vl) * 2


# ------------------------------------------------------------- templates
def template_facts(src):
    """How a Jinja chat template uses tools, with up to three lines of evidence."""
    lines = src.splitlines()
    hits = []
    for m in _BLOCK.finditer(src):
        if _TOOLS_VAR.search(_STRING.sub('""', m.group(1))):
            hits.append(src.count("\n", 0, m.start()))
    if hits:
        kind, picked = "request-level", sorted(set(hits))[:EXCERPT_LINES]
    elif _TOOLS_FIELD.search(src):
        kind = "system-message field only"
        picked = [i for i, ln in enumerate(lines) if _TOOLS_FIELD.search(ln)][:EXCERPT_LINES]
    else:
        kind, picked = "none", list(range(min(EXCERPT_LINES, len(lines))))
    return {"chars": len(src), "sha256": hashlib.sha256(src.encode()).hexdigest(), "tools_reference": kind,
            "excerpt": [lines[i].strip()[:EXCERPT_CHARS] for i in picked]}


# ----------------------------------------------------------------- bundles
def bundle_facts(path):
    zf = zipfile.ZipFile(path)
    members = {i.filename.rsplit("/", 1)[-1]: i for i in zf.infolist() if not i.is_dir()}

    def text(name):
        return json.loads(zf.read(members[name]))
    cfg, meta, tok = text("genie_config.json"), text("metadata.json"), text("tokenizer_config.json")
    bins = cfg["dialog"]["engine"]["model"]["binary"]["ctx-bins"]
    largest = max(bins, key=lambda b: members[b].file_size)
    table = meta["genie"]["chat_template"]
    vision = "vision_encoder.bin" in members or bool(meta["genie"].get("supports_vision"))
    jinja = template_facts(tok["chat_template"])
    has_tool_entry = any("tool" in k for k in table)
    # GenieX v0.8.0 sends a bundle with a vision encoder through its vision
    # pipeline, which builds the prompt from metadata.json's prefix/suffix
    # table; text bundles are rendered from the Jinja template. This rests on
    # the stage 1 prompt reconstruction (exact token counts on every request)
    # and on strings in the public release, not on anything in the bundle.
    if vision:
        construction, slot = "prefix/suffix table (vision pipeline)", has_tool_entry
        evidence = (f"vision_encoder.bin present; metadata.json table keys {sorted(table)} have no tool entry; "
                    f"the unused tokenizer_config.json template references tools: {jinja['tools_reference']}")
    else:
        construction, slot = "Jinja chat template", jinja["tools_reference"] == "request-level"
        evidence = f"tokenizer_config.json chat_template, tools reference: {jinja['tools_reference']}"
    return {"zip_bytes": os.path.getsize(path), "model_id": meta.get("model_id"),
            "qairt": meta.get("tool_versions", {}).get("qairt"),
            "context_size": cfg["dialog"]["context"]["size"], "context_lengths": meta["genie"].get("context_lengths"),
            "n_parts": len(bins), "largest_part": largest, "largest_part_bytes": members[largest].file_size,
            "vision_encoder": vision, "jinja_template": jinja, "prefix_table_keys": sorted(table),
            "prompt_construction": construction, "tool_slot": slot, "tool_slot_evidence": evidence}


# ------------------------------------------------------------------ driver
def download_log_hashes(art_dir):
    """{path relative to art_dir: sha256} from download.log lines 'OK model kind bytes sha256 path'."""
    out = {}
    try:
        lines = open(os.path.join(art_dir, "download.log")).read().splitlines()
    except OSError:
        return out
    for ln in lines:
        p = ln.split()
        if len(p) >= 6 and p[0] == "OK":
            out[os.path.relpath(os.path.expanduser(p[5]), art_dir)] = p[4]
    return out


def recorded_sha(results_dir, model, path):
    for name in ("server.json", "load_failure.json"):
        try:
            return json.load(open(os.path.join(results_dir, model, path, name)))["artifact_sha256"]
        except (OSError, KeyError):
            continue
    return None


def origin(gguf_source):
    """Coarse origin of a GGUF from campaign/v2_models.json's gguf_source."""
    return next(k for k in ("catalog", "community", "self-quantized") if gguf_source.startswith(k))


def gguf_facts(path):
    version, n_tensors, kv = gguf_metadata(path)
    tmpl = kv.get("tokenizer.chat_template")
    return {"bytes": os.path.getsize(path), "gguf_version": version, "n_tensors": n_tensors,
            "architecture": kv["general.architecture"], "size_label": kv.get("general.size_label"),
            "file_type": FILE_TYPES.get(kv.get("general.file_type"), kv.get("general.file_type")),
            "quantization_version": kv.get("general.quantization_version"),
            "imatrix": "quantize.imatrix.file" in kv,
            "context_length_trained": kv.get(f"{kv['general.architecture']}.context_length"),
            "kv_bytes_per_token_f16": kv_bytes_per_token(kv),
            "chat_template": template_facts(tmpl) if isinstance(tmpl, str) else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("artifacts_dir")
    ap.add_argument("--results", default=os.path.join(REPO, "results", "v2"))
    ap.add_argument("--out", default=os.path.join(HERE, "data", "artifacts.json"))
    a = ap.parse_args()
    art = os.path.abspath(os.path.expanduser(a.artifacts_dir))
    spec = json.load(open(os.path.join(REPO, "campaign", "v2_models.json")))
    logged = download_log_hashes(art)
    out = {"catalog_release": spec["catalog_release"], "models": {}}
    for m in spec["models"]:
        entry = {"tier": m["tier"]}
        for kind, rel, path in (("gguf", m["gguf"], "cpu"), ("bundle", m["bundle"], "compiled")):
            if not rel:
                entry[kind] = None
                continue
            facts = gguf_facts(os.path.join(art, rel)) if kind == "gguf" else bundle_facts(os.path.join(art, rel))
            sha = recorded_sha(a.results, m["name"], path)
            entry[kind] = {"file": rel, "sha256_recorded": sha,
                           "sha256_matches_download_log": (logged[rel] == sha) if rel in logged else None,
                           **facts}
        entry["gguf"]["origin"] = origin(m["gguf_source"])
        out["models"][m["name"]] = entry
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {a.out}: {len(out['models'])} models")


if __name__ == "__main__":
    main()
