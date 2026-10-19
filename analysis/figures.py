#!/usr/bin/env python3
"""Figures for the version 2 report, drawn as SVG. Standard library only.

Reads the JSON that analysis/report.py writes (analysis/out/report.json) and
writes seven SVG files to docs/figures/. Nothing is recomputed here: every
number drawn is a number in report.json, so the figures agree with the report
tables.

- coverage.svg    stack.coverage: each of the 13 models on each of the three
                  serving paths, filled by status (readable, with its pooled end
                  to end accuracy; not readable, with the reason; did not load,
                  with the failure; not offered). Rows are ordered by Q4_0 GGUF
                  size (stack.load_boundary), which puts the observed NPU
                  (llama.cpp) load boundary between two rows; the line is
                  labelled with that path, since the CPU loaded every GGUF.
- accuracy.svg    accuracy.configs: pooled end to end accuracy over 260 items
                  with the 95% item bootstrap interval, readable configurations
                  only, sorted.
- frontier.svg    speed.tradeoff: pooled accuracy against median canonical turn
                  time (log axis), with the Pareto frontier as a staircase; a
                  step that the exact McNemar test (Holm) does not separate is
                  dashed.
- turn_time.svg   speed.turn: median canonical turn time split by the median
                  per-request prefill share (server prompt time over client
                  latency); the rest is generation plus request overhead (under
                  2% of the time summed over the suite). The per-request median
                  is used rather than the suite-wide share because a few
                  runaway generations (Qwen3-4B and SmolLM2 compiled) pull the
                  suite-wide share well below that of a typical turn, which the
                  time-to-first-token probes confirm. Configurations whose
                  prompts lacked the tools are left out: their prompts were
                  under 650 tokens, against about 2,700.
- throughput.svg  speed.probes: mean time to first token over 5 repetitions at
                  five prompt lengths, for Qwen3-1.7B and Qwen3-4B-Instruct-2507.
- errors.svg      errors.taxonomy: each readable configuration's failing
                  repetitions by category, as shares of its failures. Adjacent
                  categories are merged into seven columns that keep the
                  taxonomy's precedence order; a category not drawn
                  (request_error, text_parseable) must be zero in every drawn
                  row, or the figure is not written.
- paraphrase.svg  robustness.styles: canonical and per-style accuracy for the
                  functional pool, each configuration and the pool mean with its
                  95% item bootstrap interval.

One colour and one marker shape per serving path throughout (a palette that
passes colour-vision-deficiency separation checks); text is never coloured by
series. Fonts are the platform sans-serif; text widths are estimated from
DejaVu Sans metrics, the widest common fallback, so labels do not collide when
the fallback is used.

Usage:
  python3 analysis/figures.py [report.json] [out_dir]
"""
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))

W = 720
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
PATHS = ("cpu", "npu_llama", "compiled")
COLOUR = {"cpu": "#2a78d6", "npu_llama": "#199e70", "compiled": "#eb6834"}
SHAPE = {"cpu": "circle", "npu_llama": "square", "compiled": "diamond"}
STATUS_FILL = {"readable": "#d3ecd3", "unreadable": "#fbe6b0", "load failure": "#f4cfcc", "not offered": "#f1f0ec"}
STATUS_NAME = {"readable": "Readable (end to end accuracy)", "unreadable": "Not readable as served",
               "load failure": "Did not load", "not offered": "Not offered on this path"}
HEAT_LO, HEAT_HI, HEAT_NONE = "#f3f1fb", "#3a2d8c", "#fafaf8"


# ------------------------------------------------------------------ drawing
def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tw(s, size, bold=False):
    """Estimated rendered width of s in px (DejaVu Sans advance widths)."""
    em = 0
    for ch in str(s):
        if ch in "iljI.,:;'!|":
            em += 0.30
        elif ch in " ()[]-/ft":
            em += 0.38
        elif ch in "mwMW%":
            em += 0.92
        elif ch.isupper():
            em += 0.72
        else:
            em += 0.62
    return em * size * (1.1 if bold else 1)


def text(x, y, s, size=12, fill=INK, anchor="start", weight=None, halo=False):
    """Text; a halo in the surface colour keeps it legible where it crosses lines."""
    w = f' font-weight="{weight}"' if weight else ""
    h = f' stroke="{SURFACE}" stroke-width="3" stroke-linejoin="round" paint-order="stroke"' if halo else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}"{w}{h}>'
            f'{esc(s)}</text>')


def line(x1, y1, x2, y2, stroke=GRID, width=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
            f'stroke-width="{width}"{d}/>')


def rect(x, y, w, h, fill, tip=None, rx=0):
    t = f"<title>{esc(tip)}</title>" if tip else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}">{t}</rect>'


def marker(path, x, y, r=4.5, tip=None, ring=True):
    """The path's marker in its colour, the three shapes of about equal area, with
    a surface ring so overlapping markers stay legible."""
    t = f"<title>{esc(tip)}</title>" if tip else ""
    paint = f'fill="{COLOUR[path]}"' + (f' stroke="{SURFACE}" stroke-width="1.5"' if ring else "")
    if SHAPE[path] == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r + 0.3:.1f}" {paint}>{t}</circle>'
    if SHAPE[path] == "square":
        s = r * 1.85
        return f'<rect x="{x - s / 2:.1f}" y="{y - s / 2:.1f}" width="{s:.1f}" height="{s:.1f}" {paint}>{t}</rect>'
    d = r * 1.4
    return (f'<path d="M{x:.1f},{y - d:.1f} L{x + d:.1f},{y:.1f} L{x:.1f},{y + d:.1f} L{x - d:.1f},{y:.1f} Z" '
            f'{paint}>{t}</path>')


def document(h, title, desc, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h:.0f}" viewBox="0 0 {W} {h:.0f}" '
            f'role="img" font-family="{FONT}">\n<title>{esc(title)}</title>\n<desc>{esc(desc)}</desc>\n'
            f'<rect width="{W}" height="{h:.0f}" fill="{SURFACE}"/>\n' + "\n".join(body) + "\n</svg>\n")


def mix(a, b, t):
    """Colour a moved a fraction t of the way to colour b, in sRGB."""
    pa = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    pb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(pa, pb))


def luminance(c):
    def ch(v):
        v /= 255
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (ch(int(c[i:i + 2], 16)) for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def wrap(s, width, size):
    lines, cur = [], ""
    for word in s.split():
        if cur and tw(cur + " " + word, size) > width:
            lines.append(cur)
            cur = word
        else:
            cur = (cur + " " + word).strip()
    return lines + [cur]


def dodge(ys, gap, hi=None):
    """Label baselines for ys in ascending order, at least gap apart, shifted up
    together if the last would pass hi."""
    out = []
    for y in sorted(ys):
        out.append(max(y, out[-1] + gap) if out else y)
    over = max(0, out[-1] - hi) if hi is not None and out else 0
    return [y - over for y in out]


def key(x, y, names):
    """A one-row key of path markers and names."""
    out = []
    for p in PATHS:
        out += [marker(p, x + 5, y - 4), text(x + 15, y, names[p], 11, INK2)]
        x += 15 + tw(names[p], 11) + 18
    return out


def lin(d0, d1, r0, r1):
    return lambda v: r0 + (v - d0) / (d1 - d0) * (r1 - r0)


def logscale(d0, d1, r0, r1):
    return lambda v: r0 + (math.log10(v) - math.log10(d0)) / (math.log10(d1) - math.log10(d0)) * (r1 - r0)


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}"


# ------------------------------------------------------------------ data
def names(rep):
    """Display names of models and paths, as the accuracy module records them."""
    cfgs = rep["accuracy"]["configs"]
    return ({c["model"]: c["model_name"] for c in cfgs}, {c["path"]: c["path_name"] for c in cfgs})


def short(name):
    """A model name without its instruction-tuning tag, for point labels."""
    return re.sub(r"-(Instruct|instruct|it)\b", "", name)


def readable(rep):
    """Readable configurations, best first (ties by name)."""
    rows = [c for c in rep["accuracy"]["configs"] if c["readable"]]
    return sorted(rows, key=lambda c: (-c["e2e"], c["model_name"], PATHS.index(c["path"])))


def key_of(c):
    return f"{c['model']}/{c['path']}"


# ------------------------------------------------------------------ 1. coverage
def fig_coverage(rep):
    mnames, pnames = names(rep)
    cells = {(c["model"], c["path"]): c for c in rep["stack"]["coverage"]["cells"]}
    acc = {(c["model"], c["path"]): c["e2e"] for c in rep["accuracy"]["configs"]}
    size = {r["model"]: r["gguf_bytes"] for r in rep["stack"]["load_boundary"]["rows"]}
    boundary = rep["stack"]["load_boundary"]
    loaded = {r["model"]: r["loaded"] for r in boundary["rows"]}
    models = sorted({m for m, _ in cells}, key=lambda m: (size[m], m))
    x0, xs, xp, cw, rh, top = 16, 230, 300, 136, 34, 84
    # A band between the rows on either side of the boundary holds its label.
    n_loaded, band = sum(loaded.values()), 14 if boundary["one_gib_between"] else 0
    body = []
    x = x0
    for st in STATUS_FILL:
        body += [rect(x, 12, 14, 14, STATUS_FILL[st], rx=2), text(x + 20, 23.5, STATUS_NAME[st], 11, INK2)]
        x += 20 + tw(STATUS_NAME[st], 11) + 18
    body += [text(x0, top - 12, "Model", 11, MUTED), text(xs + 56, top - 26, "Q4_0 GGUF", 11, MUTED, "end"),
             text(xs + 56, top - 12, "size, GiB", 11, MUTED, "end")]
    for j, p in enumerate(PATHS):
        cx = xp + j * cw + cw / 2
        wname = tw(pnames[p], 12)
        body += [marker(p, cx - wname / 2 - 5, top - 16), text(cx + 6, top - 12, pnames[p], 12, INK, "middle")]
    for i, m in enumerate(models):
        y = top + i * rh + (band if i >= n_loaded else 0)
        body += [text(x0, y + rh / 2 + 4, mnames[m], 12),
                 text(xs + 56, y + rh / 2 + 4, f"{size[m] / 2 ** 30:.2f}", 12, INK2, "end")]
        for j, p in enumerate(PATHS):
            c = cells[(m, p)]
            st = c["status"]
            cx, cy = xp + j * cw, y + 1
            tip = f"{mnames[m]}, {pnames[p]}: {STATUS_NAME[st].lower()}"
            tip += f", {c['reason']}" if c.get("reason") else ""
            body.append(rect(cx + 1, cy, cw - 2, rh - 2, STATUS_FILL[st], tip, rx=3))
            mid = cx + cw / 2
            if st == "readable":
                body.append(text(mid, cy + rh / 2 + 3.5, f"{pct(acc[(m, p)])}%", 13, INK, "middle", 600))
            elif st == "not offered":
                body.append(text(mid, cy + rh / 2 + 3, "not offered", 11, MUTED, "middle"))
            else:
                head = "not readable" if st == "unreadable" else "did not load"
                body += [text(mid, cy + 13, head, 11, INK, "middle"),
                         text(mid, cy + 26, c["reason"], 10.5, INK2, "middle")]
    # The observed NPU (llama.cpp) boundary, drawn where every GGUF above it
    # loaded and every one below did not, which the stack module checks. The
    # label names the path, so the line is not read as a limit on every path.
    if boundary["one_gib_between"]:
        y = top + n_loaded * rh + band / 2
        lab = f"{pnames['npu_llama']} load boundary"
        body += [line(x0, y, xp + 3 * cw, y, INK2, 1.2, "5 3"),
                 rect(x0 - 2, y - 7, 18 + tw(lab, 10.5, True), 14, SURFACE), marker("npu_llama", x0 + 4, y, 3.5),
                 text(x0 + 13, y + 4, lab, 10.5, INK, weight=600),
                 rect(xs + 6, y - 7, 54, 14, SURFACE), text(xs + 56, y + 4, "1 GiB", 10.5, INK, "end", 600)]
    h = top + len(models) * rh + band + 14
    return document(h, "Serving-path coverage",
                    "Status of each model on each serving path, with pooled end to end accuracy for readable "
                    "configurations; rows ordered by GGUF size.", body)


# ------------------------------------------------------------------ 2. accuracy
def fig_accuracy(rep):
    _, pnames = names(rep)
    rows = readable(rep)
    x0, xpath, px0, px1, rh, top = 16, 196, 330, 664, 22, 40
    sx = lin(0, 1, px0, px1)
    bottom = top + len(rows) * rh
    body = [text(x0, top - 14, "Model", 11, MUTED), text(xpath, top - 14, "Serving path", 11, MUTED)]
    for v in range(0, 101, 20):
        body += [line(sx(v / 100), top - 6, sx(v / 100), bottom + 2),
                 text(sx(v / 100), bottom + 16, f"{v}", 11, MUTED, "middle")]
    body.append(text((px0 + px1) / 2, bottom + 34, "End to end accuracy, % of 260 items (95% bootstrap interval)",
                     12, INK2, "middle"))
    for i, c in enumerate(rows):
        y = top + i * rh + rh / 2
        lo, hi = c["e2e_ci_item"]
        tip = f"{c['model_name']}, {c['path_name']}: {pct(c['e2e'])}% ({pct(lo)}-{pct(hi)})"
        col = COLOUR[c["path"]]
        body += [text(x0, y + 4, c["model_name"], 12), marker(c["path"], xpath + 5, y),
                 text(xpath + 15, y + 4, pnames[c["path"]], 11, INK2),
                 line(sx(lo), y, sx(hi), y, col, 1.5), line(sx(lo), y - 4, sx(lo), y + 4, col, 1.5),
                 line(sx(hi), y - 4, sx(hi), y + 4, col, 1.5),
                 marker(c["path"], sx(c["e2e"]), y, tip=tip, ring=False),
                 text(sx(hi) + 8, y + 4, pct(c["e2e"]), 11, INK)]
    return document(bottom + 46, "End to end accuracy of readable configurations",
                    "Pooled end to end accuracy with 95% item bootstrap intervals for the 19 readable "
                    "configurations, sorted.", body)


# ------------------------------------------------------------------ 3. frontier
def overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def label_box(x, y, s, size, anchor):
    w = tw(s, size)
    x0 = x if anchor == "start" else x - w if anchor == "end" else x - w / 2
    return (x0 - 1, y - size * 0.8, x0 + w + 1, y + size * 0.25)


def candidates(hw, hh):
    """Label offsets around an anchor box of half-width hw and half-height hh,
    nearest first; the second list is far enough to need a leader line."""
    near = [(hw + 9, 4, "start"), (-hw - 9, 4, "end"), (0, -hh - 9, "middle"), (0, hh + 17, "middle"),
            (hw + 7, -hh - 7, "start"), (-hw - 7, -hh - 7, "end"), (hw + 7, hh + 14, "start"),
            (-hw - 7, hh + 14, "end")]
    far = [(dx, dy, a) for d in (16, 26, 38) for dx, dy, a in
           ((hw + d, -hh - d, "start"), (-hw - d, -hh - d, "end"), (hw + d, hh + d + 6, "start"),
            (-hw - d, hh + d + 6, "end"), (hw + d + 6, 4, "start"), (-hw - d - 6, 4, "end"),
            (0, -hh - d, "middle"), (0, hh + d + 10, "middle"))]
    return near, far


def place(points, taken, bounds, size=11):
    """Greedy label placement. points: (x, y, half-width, half-height, label) in
    priority order. Each label goes to the first candidate position, nearest
    first, that overlaps no marker, line or label placed so far; far positions
    get a leader line."""
    out = []
    for x, y, hw, hh, s in points:
        near, far = candidates(hw, hh)
        for dx, dy, a in near + far:
            box = label_box(x + dx, y + dy, s, size, a)
            inside = box[0] >= bounds[0] and box[2] <= bounds[2] and box[1] >= bounds[1] and box[3] <= bounds[3]
            if inside and not any(overlaps(box, t) for t in taken):
                taken.append(box)
                leader = None
                if (dx, dy, a) in far:
                    leader = (x, y, min(max(x, box[0]), box[2]), min(max(y, box[1]), box[3]))
                out.append((x + dx, y + dy, s, a, leader))
                break
        else:
            raise ValueError(f"no room for label {s!r}")
    return out


def fig_frontier(rep):
    mnames, pnames = names(rep)
    tr = rep["speed"]["tradeoff"]
    pts = {p["config"]: p for p in tr["points"]}
    px0, px1, py0, py1 = 58, 704, 44, 444
    sx, sy = logscale(2, 100, px0, px1), lin(0, 1, py1, py0)
    body = []
    for v in (2, 3, 5, 10, 20, 30, 50, 100):
        body += [line(sx(v), py0, sx(v), py1), text(sx(v), py1 + 16, f"{v}", 11, MUTED, "middle")]
    for v in range(0, 101, 20):
        body += [line(px0, sy(v / 100), px1, sy(v / 100)), text(px0 - 8, sy(v / 100) + 4, f"{v}", 11, MUTED, "end")]
    body += [text((px0 + px1) / 2, py1 + 36, "Median agent turn time, s (log scale; 65 canonical cases)", 12, INK2,
                  "middle"),
             f'<text transform="translate(16,{(py0 + py1) / 2:.1f}) rotate(-90)" font-size="12" fill="{INK2}" '
             f'text-anchor="middle">End to end accuracy, % of 260 items</text>']
    sig = {(s["faster"], s["slower"]): s["significant"] for s in tr["steps"]}
    front = sorted(tr["frontier"], key=lambda k: pts[k]["t"])
    taken = []
    for a, b in zip(front, front[1:]):
        xa, ya, xb, yb = sx(pts[a]["t"]), sy(pts[a]["e2e"]), sx(pts[b]["t"]), sy(pts[b]["e2e"])
        dash = None if sig[(a, b)] else "4 3"
        body += [line(xa, ya, xb, ya, INK2, 1.5, dash), line(xb, ya, xb, yb, INK2, 1.5, dash)]
        taken += [(xa, ya - 2, xb, ya + 2), (xb - 2, yb, xb + 2, ya)]
    # Label the frontier on its longest flat step, above the line.
    a, b = max(zip(front, front[1:]), key=lambda ab: sx(pts[ab[1]]["t"]) - sx(pts[ab[0]]["t"]))
    fx, fy = (sx(pts[a]["t"]) + sx(pts[b]["t"])) / 2, sy(pts[a]["e2e"]) - 7
    body.append(text(fx, fy, "Pareto frontier", 11, INK2, "middle", 600))
    taken.append(label_box(fx, fy, "Pareto frontier", 11, "middle"))
    kx, ky = px1 - 250, py1 - 64
    for i, p in enumerate(PATHS):
        body += [marker(p, kx + 6, ky + i * 17 - 4), text(kx + 16, ky + i * 17, pnames[p], 11, INK2)]
    body += [line(kx, ky + 3 * 17 - 4, kx + 14, ky + 3 * 17 - 4, INK2, 1.5, "4 3"),
             text(kx + 20, ky + 3 * 17, "frontier step not significant", 11, INK2)]
    taken.append((kx - 4, ky - 14, px1, ky + 3 * 17 + 4))
    order = sorted(pts, key=lambda k: (k not in front, -pts[k]["e2e"]))
    pos = {k: (sx(pts[k]["t"]), sy(pts[k]["e2e"])) for k in order}
    taken += [(x - 6, y - 6, x + 6, y + 6) for x, y in pos.values()]
    # Points of one model that nearly touch share a label.
    anchors, done = [], set()
    for k in order:
        if k in done:
            continue
        m = k.split("/")[0]
        grp = [j for j in order if j.split("/")[0] == m and j not in done and math.dist(pos[j], pos[k]) < 24]
        done.update(grp)
        xs, ys = [pos[j][0] for j in grp], [pos[j][1] for j in grp]
        anchors.append(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (max(xs) - min(xs)) / 2,
                        (max(ys) - min(ys)) / 2, short(mnames[m])))
    labels = place(anchors, taken, (px0 + 2, py0 - 30, px1, py1 - 2))
    for k in order:
        m, path = k.split("/")
        tip = f"{mnames[m]}, {pnames[path]}: {pct(pts[k]['e2e'])}%, {pts[k]['t']:.2f} s"
        body.append(marker(path, *pos[k], 5, tip))
    for x, y, s, a, leader in labels:
        if leader:
            body.append(line(*leader, MUTED, 0.8))
        body.append(text(x, y, s, 11, INK, a))
    return document(py1 + 48, "Accuracy against agent turn time",
                    "Pooled end to end accuracy against median canonical turn time on a log axis for the 19 "
                    "readable configurations, with the Pareto frontier.", body)


# ------------------------------------------------------------------ 4. turn time
def fig_turn_time(rep):
    mnames, pnames = names(rep)
    seen = {k: t for k, t in rep["speed"]["turn"].items() if t["tools_seen"]}
    groups = {}
    for k in seen:
        m, p = k.split("/")
        groups.setdefault(m, []).append(p)
    order = sorted(groups, key=lambda m: (-max(seen[f"{m}/{p}"]["median_s"] for p in groups[m]), m))
    readable_keys = {key_of(c) for c in rep["accuracy"]["configs"] if c["readable"]}
    x0, xpath, px0, px1, rh, gh, top = 16, 26, 142, 580, 19, 20, 8
    tmax = 10 * math.ceil(max(t["median_s"] for t in seen.values()) / 10)
    sx = lin(0, tmax, px0, px1)
    bottom = top + len(order) * gh + len(seen) * rh + 6
    body = []
    for v in range(0, tmax + 1, 10):
        body += [line(sx(v), top, sx(v), bottom), text(sx(v), bottom + 15, f"{v}", 11, MUTED, "middle")]
    body.append(text((px0 + px1) / 2, bottom + 33,
                     "Median agent turn time, s (65 canonical cases, about 2,700 prompt tokens)", 12, INK2, "middle"))
    y = top
    for m in order:
        y += gh
        body.append(text(x0, y - 5, mnames[m], 12, INK, weight=600))
        if not any(f"{m}/{p}" in readable_keys for p in groups[m]):
            body.append(text(x0 + tw(mnames[m], 12, True) + 6, y - 5, "(replies not readable)", 11, MUTED))
        for p in sorted(groups[m], key=PATHS.index):
            t = seen[f"{m}/{p}"]
            cy = y + rh / 2
            share = t["prefill_share_request_median"]
            total, pre = t["median_s"], t["median_s"] * share
            tip = f"{mnames[m]}, {pnames[p]}: {total:.2f} s, prompt reading {pct(share)}%"
            body += [marker(p, xpath + 4, cy, 4), text(xpath + 14, cy + 4, pnames[p], 11, INK2),
                     rect(sx(0), cy - 7, sx(total) - sx(0), 14, mix(COLOUR[p], SURFACE, 0.6), tip),
                     rect(sx(0), cy - 7, sx(pre) - sx(0), 14, COLOUR[p], tip),
                     line(sx(pre) + 0.5, cy - 7, sx(pre) + 0.5, cy + 7, SURFACE, 1.5),
                     text(sx(total) + 6, cy + 4, f"{total:.2f} s, prompt {pct(share)}%", 11, INK)]
            if y == top + gh:
                # The first bar is the longest; it names both parts for every bar.
                gx = (sx(pre) + sx(total)) / 2
                body += [text(sx(0) + 6, cy + 4, "prompt reading (prefill)", 11, SURFACE, weight=600),
                         text(sx(total), y - 5, "answer generation (decode)", 11, INK2, "end"),
                         line(gx, y - 2, gx, cy - 7, INK2, 1)]
            y += rh
    return document(bottom + 44, "Agent turn time split into prompt reading and generation",
                    "Median canonical agent turn time per configuration whose prompts carried the tools, split "
                    "by the median per-request share of time the server spent reading the prompt.", body)


# ------------------------------------------------------------------ 5. throughput
def fig_throughput(rep, models=("qwen3-1.7b", "qwen3-4b-2507")):
    mnames, pnames = names(rep)
    probes = rep["speed"]["probes"]
    longest = rep["speed"]["ttft_longest"]
    pw, gap, px0, py0, py1 = 196, 128, 64, 66, 346
    sy = logscale(0.1, 100, py1, py0)
    body = key(16, 20, pnames)
    body.append(f'<text transform="translate(18,{(py0 + py1) / 2:.1f}) rotate(-90)" font-size="12" fill="{INK2}" '
                f'text-anchor="middle">Time to first token, s (log scale)</text>')
    for v, lab in ((0.1, "0.1"), (0.3, "0.3"), (1, "1"), (3, "3"), (10, "10"), (30, "30"), (100, "100")):
        body.append(text(px0 - 8, sy(v) + 4, lab, 11, MUTED, "end"))
    for i, m in enumerate(models):
        x0 = px0 + i * (pw + gap)
        sx = lin(0, 3000, x0, x0 + pw)
        body.append(text(x0, py0 - 12, mnames[m], 12, INK, weight=600))
        for v in (0.1, 0.3, 1, 3, 10, 30, 100):
            body.append(line(x0, sy(v), x0 + pw, sy(v)))
        for v in range(0, 3001, 1000):
            body += [line(sx(v), py0, sx(v), py1), text(sx(v), py1 + 16, f"{v:,}", 11, MUTED, "middle")]
        body.append(text(x0 + pw / 2, py1 + 34, "Prompt length, tokens", 12, INK2, "middle"))
        ends = []
        for p in PATHS:
            k = f"{m}/{p}"
            if k not in probes:
                continue
            rows = probes[k]["agent"]
            xy = [(sx(r["prompt_tokens"]), sy(r["ttft_s"]["mean"])) for r in rows]
            body.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in xy)}" fill="none" '
                        f'stroke="{COLOUR[p]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
            for r, (x, y) in zip(rows, xy):
                tip = f"{mnames[m]}, {pnames[p]}: {r['ttft_s']['mean']:.2f} s at {r['prompt_tokens']:,} tokens"
                body.append(marker(p, x, y, 3.5, tip))
            lg = longest[k]
            lab = f"{lg['mean']:.2f} s" + (f", {lg['speedup_vs_cpu']:.1f}x" if lg["speedup_vs_cpu"] else "")
            ends.append((xy[-1][1] + 4, lab))
        ends.sort()
        body += [text(x0 + pw + 8, y, lab, 11, INK) for y, (_, lab) in zip(dodge([e[0] for e in ends], 14), ends)]
        missing = [p for p in PATHS if f"{m}/{p}" not in probes]
        for p in missing:
            body.append(text(x0 + pw - 4, py1 - 10, f"{pnames[p]} did not load", 11, MUTED, "end"))
    body.append(text(W / 2, py1 + 58, "End labels give the time at about 2,800 prompt tokens and the speed-up over "
                     "the CPU.", 11, INK2, "middle"))
    return document(py1 + 70, "Time to first token against prompt length",
                    "Mean time to first token over 5 repetitions at five prompt lengths, for two models on each "
                    "serving path they loaded on; end labels give the value at about 2,800 tokens and the speed-up "
                    "over the CPU.", body)


# ------------------------------------------------------------------ 6. errors
ERROR_GROUPS = (
    ("No structured call", (
        ("Call in text, invalid JSON", ("text_malformed",)),
        ("Call in text, other form", ("text_other_shape", "text_pseudo")),
        ("Prose, no call", ("prose",)))),
    ("Structured call with an error", (
        ("Command or other name as function", ("command_as_tool", "other_tool")),
        ("Wrong tool", ("wrong_tool",)),
        ("Wrong command", ("wrong_command",)),
        ("Wrong device or parameters", ("wrong_device", "wrong_params")))),
)


def fig_errors(rep):
    _, pnames = names(rep)
    tax = {f"{c['model']}/{c['path']}": c["pooled"] for c in rep["errors"]["taxonomy"]["configs"]}
    cols = [(lab, cats) for _, grp in ERROR_GROUPS for lab, cats in grp]
    order = [c for _, cats in cols for c in cats]
    cats_all = rep["errors"]["taxonomy"]["categories"]
    # Columns must follow the taxonomy's precedence, since the caption says a
    # failure takes the first column that applies.
    if order != [c for c in cats_all if c in order]:
        raise ValueError("error columns out of taxonomy order")
    rows = readable(rep)
    missing = [c for c in cats_all if c not in order]
    if any(tax[key_of(r)]["counts"][c] for r in rows for c in missing):
        raise ValueError(f"error categories not drawn: {missing}")
    n_reps = {tax[key_of(r)]["n_reps"] for r in rows}
    if len(n_reps) != 1:
        raise ValueError(f"readable configurations differ in repetitions: {sorted(n_reps)}")
    x0, xpath, cx0, cw, rh = 16, 170, 280, 54, 22
    head = [wrap(lab, cw - 4, 10.5) for lab, _ in cols]
    top = 34 + 13 * max(len(h) for h in head) + 8
    body = []
    j = 0
    for gname, grp in ERROR_GROUPS:
        gx0, gx1 = cx0 + j * cw + 2, cx0 + (j + len(grp)) * cw - 2
        body += [text((gx0 + gx1) / 2, 16, gname, 11, INK, "middle", 600), line(gx0, 22, gx1, 22, AXIS, 1)]
        j += len(grp)
    for j, h in enumerate(head):
        for i, s in enumerate(h):
            body.append(text(cx0 + j * cw + cw / 2, 36 + i * 13, s, 10.5, INK2, "middle"))
    fx = cx0 + len(cols) * cw + 6
    body += [text(fx + 34, 36, "Failing", 10.5, INK2, "end"), text(fx + 34, 49, "reps", 10.5, INK2, "end"),
             text(fx + 34, 62, f"of {n_reps.pop()}", 10.5, INK2, "end"),
             text(x0, top - 8, "Model", 11, MUTED), text(xpath, top - 8, "Serving path", 11, MUTED)]
    for i, c in enumerate(rows):
        y = top + i * rh
        t = tax[key_of(c)]
        body += [text(x0, y + rh / 2 + 4, c["model_name"], 11.5), marker(c["path"], xpath + 5, y + rh / 2, 4),
                 text(xpath + 14, y + rh / 2 + 4, pnames[c["path"]], 11, INK2),
                 text(fx + 34, y + rh / 2 + 4, f"{t['n_fail']}", 11, INK, "end")]
        for j, (lab, cats) in enumerate(cols):
            n = sum(t["counts"][k] for k in cats)
            share = n / t["n_fail"] if n else 0
            # A power below 1 lifts small shares off the empty-cell colour.
            fill = mix(HEAT_LO, HEAT_HI, share ** 0.8) if n else HEAT_NONE
            tip = f"{c['model_name']}, {c['path_name']}: {lab.lower()}, {n} of {t['n_fail']} failing reps"
            body.append(rect(cx0 + j * cw + 1, y + 1, cw - 2, rh - 2, fill, tip, rx=2))
            if n:
                ink = SURFACE if luminance(fill) < 0.3 else INK
                body.append(text(cx0 + j * cw + cw / 2, y + rh / 2 + 4, f"{share * 100:.0f}" if share >= 0.005
                                 else "<1", 11, ink, "middle"))
    bottom = top + len(rows) * rh
    body.append(text(cx0 + len(cols) * cw / 2, bottom + 18,
                     "Share of the configuration's failing repetitions, %; a blank cell is none", 11, INK2,
                     "middle"))
    return document(bottom + 30, "Composition of failures",
                    "Failing repetitions of each readable configuration by error category, as a percentage of "
                    "its failing repetitions.", body)


# ------------------------------------------------------------------ 7. paraphrase
STYLE_ORDER = (("canonical", "Canonical"), ("A", "Casual (A)"), ("B", "Polite (B)"), ("C", "Situational (C)"))


def fig_paraphrase(rep):
    mnames, pnames = names(rep)
    rob = rep["robustness"]
    pool = rob["pools"]["functional"]
    acc = {key_of(c): c for c in rep["accuracy"]["configs"]}
    px0, px1, py0, py1, lx = 58, 420, 24, 444, 470
    sx = {s: px0 + 30 + i * (px1 - px0 - 60) / 3 for i, (s, _) in enumerate(STYLE_ORDER)}
    sy = lin(0, 1, py1, py0)
    body = []
    for v in range(0, 101, 20):
        body += [line(px0, sy(v / 100), px1, sy(v / 100)), text(px0 - 8, sy(v / 100) + 4, f"{v}", 11, MUTED, "end")]
    for s, lab in STYLE_ORDER:
        body.append(text(sx[s], py1 + 18, lab, 12, INK2, "middle"))
    body += [text((px0 + px1) / 2, py1 + 38, "Phrasing of the request (65 requests each)", 12, INK2, "middle"),
             f'<text transform="translate(16,{(py0 + py1) / 2:.1f}) rotate(-90)" font-size="12" fill="{INK2}" '
             f'text-anchor="middle">End to end accuracy, %</text>']

    def values(k):
        st = rob["styles"]["configs"][k]
        return [acc[k]["canonical"]["e2e"]] + [st[s]["acc"] for s, _ in STYLE_ORDER[1:]]

    ends = []
    for k in pool:
        m, p = k.split("/")
        v = values(k)
        pts = " ".join(f"{sx[s]:.1f},{sy(x):.1f}" for (s, _), x in zip(STYLE_ORDER, v))
        tip = f"{mnames[m]}, {pnames[p]}: " + ", ".join(f"{lab} {pct(x)}%" for (_, lab), x in zip(STYLE_ORDER, v))
        body.append(f'<polyline points="{pts}" fill="none" stroke="{COLOUR[p]}" stroke-opacity="0.75" '
                    f'stroke-width="1.5" stroke-linejoin="round"><title>{esc(tip)}</title></polyline>')
        ends.append((sy(v[-1]), f"{short(mnames[m])}, {pnames[p]}", p))
    sp = rob["styles"]["pool"]
    pv = [sp[s]["acc"] for s, _ in STYLE_ORDER]
    body.append(f'<polyline points="{" ".join(f"{sx[s]:.1f},{sy(x):.1f}" for (s, _), x in zip(STYLE_ORDER, pv))}" '
                f'fill="none" stroke="{INK}" stroke-width="2.5" stroke-linejoin="round"/>')
    for s, lab in STYLE_ORDER:
        lo, hi = sp[s]["ci"]
        x = sx[s]
        tip = f"Mean of {len(pool)} configurations, {lab}: {pct(sp[s]['acc'])}% ({pct(lo)}-{pct(hi)})"
        body += [line(x, sy(lo), x, sy(hi), INK, 2), line(x - 4, sy(lo), x + 4, sy(lo), INK, 1.5),
                 line(x - 4, sy(hi), x + 4, sy(hi), INK, 1.5),
                 f'<circle cx="{x:.1f}" cy="{sy(sp[s]["acc"]):.1f}" r="5" fill="{INK}" stroke="{SURFACE}" '
                 f'stroke-width="1.5"><title>{esc(tip)}</title></circle>',
                 text(x + 9, sy(sp[s]["acc"]) - 7, pct(sp[s]["acc"]), 11.5, INK, weight=600, halo=True)]
    ends.append((sy(pv[-1]), f"Mean of {len(pool)} configurations", None))
    ends.sort(key=lambda e: e[0])
    for y, (y_line, lab, p) in zip(dodge([e[0] for e in ends], 14, py1 - 2), ends):
        body.append(line(sx["C"] + 6, y_line, lx - 4, y, MUTED, 0.6))
        if p:
            body += [marker(p, lx + 4, y, 3.5), text(lx + 13, y + 4, lab, 11, INK2)]
        else:
            body.append(text(lx + 13, y + 4, lab, 11, INK, weight=600))
    return document(py1 + 50, "Accuracy by phrasing style",
                    "Canonical and per-style paraphrase accuracy for the 11 functional configurations, each as a "
                    "line in its path's colour, with the pool mean and its 95% item bootstrap interval.", body)


# ------------------------------------------------------------------ main
FIGURES = (("coverage", fig_coverage), ("accuracy", fig_accuracy), ("frontier", fig_frontier),
           ("turn_time", fig_turn_time), ("throughput", fig_throughput), ("errors", fig_errors),
           ("paraphrase", fig_paraphrase))


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "analysis", "out", "report.json")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "docs", "figures")
    with open(src) as f:
        rep = json.load(f)
    os.makedirs(out, exist_ok=True)
    for name, fig in FIGURES:
        path = os.path.join(out, f"{name}.svg")
        with open(path, "w") as f:
            f.write(fig(rep))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
