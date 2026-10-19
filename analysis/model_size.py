#!/usr/bin/env python3
"""Parameter counts and weight sizes of the version 2 GGUF files, read from
their headers. Standard library only.

The speed analysis fits throughput against model size, and the only reliable
source of a model's parameter count is its GGUF header, which a reader of the
repository does not have. This script reads the header of each GGUF listed in
campaign/v2_models.json from a local artifacts directory and writes the counts
to analysis/data/model_size.json, which analysis/report_speed.py reads.

Per model it records the total parameter count, the weight bytes, and the
"non-lookup" parameter count: every weight except tables that are gathered one
row per token rather than multiplied. The input embedding (token_embd) is a
lookup when the model has a separate output projection and a multiplied weight
when the output projection is tied to it; Gemma's per-layer embedding tables
(per_layer_token_embd) are always lookups. Non-lookup bytes are a first-order
estimate of the weight traffic of one decode step.

Usage:
  python3 analysis/model_size.py <artifacts_dir> [--out analysis/data/model_size.json]
"""
import argparse
import json
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))

# ggml tensor type id -> (elements per block, bytes per block)
GGML_BLOCKS = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
    10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
    16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110),
    22: (256, 82), 23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
    29: (256, 56), 30: (1, 2), 34: (256, 54), 35: (256, 66), 39: (32, 17),
}
# GGUF metadata value type id -> struct format (8 = string, 9 = array)
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def _read(fh, fmt):
    return struct.unpack("<" + fmt, fh.read(struct.calcsize(fmt)))[0]


def _string(fh):
    return fh.read(_read(fh, "Q")).decode("utf-8", errors="replace")


def _value(fh, vtype):
    if vtype == 8:
        return _string(fh)
    if vtype == 9:
        etype, n = _read(fh, "I"), _read(fh, "Q")
        if etype in SCALAR:
            fh.seek(n * struct.calcsize(SCALAR[etype]), 1)
        else:
            for _ in range(n):
                _value(fh, etype)
        return None
    return _read(fh, SCALAR[vtype])


def read_header(path):
    """(metadata, tensors) of a GGUF file; arrays in the metadata are skipped."""
    with open(path, "rb") as fh:
        if fh.read(4) != b"GGUF":
            raise ValueError(f"{os.path.basename(path)} is not a GGUF file")
        _version, n_tensors, n_kv = _read(fh, "I"), _read(fh, "Q"), _read(fh, "Q")
        meta = {}
        for _ in range(n_kv):
            key = _string(fh)
            meta[key] = _value(fh, _read(fh, "I"))
        tensors = []
        for _ in range(n_tensors):
            name = _string(fh)
            dims = [_read(fh, "Q") for _ in range(_read(fh, "I"))]
            ttype = _read(fh, "I")
            _read(fh, "Q")  # data offset
            n = 1
            for d in dims:
                n *= d
            per_block, block_bytes = GGML_BLOCKS[ttype]
            tensors.append({"name": name, "n": n, "bytes": n // per_block * block_bytes})
    return meta, tensors


def summarise(path):
    meta, tensors = read_header(path)
    tied = not any(t["name"] == "output.weight" for t in tensors)
    lookup = [t for t in tensors if "per_layer_token_embd" in t["name"]
              or (t["name"] == "token_embd.weight" and not tied)]
    arch = meta.get("general.architecture")
    return {
        "gguf": os.path.basename(path),
        "architecture": arch,
        "block_count": meta.get(f"{arch}.block_count"),
        "params_total": sum(t["n"] for t in tensors),
        "params_non_lookup": sum(t["n"] for t in tensors) - sum(t["n"] for t in lookup),
        "weight_bytes": sum(t["bytes"] for t in tensors),
        "weight_bytes_non_lookup": sum(t["bytes"] for t in tensors) - sum(t["bytes"] for t in lookup),
        "output_tied": tied,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("artifacts_dir")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "model_size.json"))
    a = ap.parse_args()
    spec = json.load(open(os.path.join(REPO, "campaign", "v2_models.json")))
    out = {m["name"]: summarise(os.path.join(a.artifacts_dir, m["gguf"])) for m in spec["models"]}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    for name, s in out.items():
        print(f"{name:18} {s['architecture']:14} {s['params_total'] / 1e9:6.3f} B total, "
              f"{s['params_non_lookup'] / 1e9:6.3f} B non-lookup, {s['weight_bytes'] / 1e9:5.2f} GB")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
