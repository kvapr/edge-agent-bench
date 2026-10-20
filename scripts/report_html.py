#!/usr/bin/env python3
"""Build the HTML version of the report, docs/report.html, from docs/REPORT.md.

The page keeps the whole report: every section, the figures (embedded as data
URIs), the request-flow diagram, every table and the list of sources, with a
contents rail and links for the cross-references ("Section 5.2", "Figure 3",
"Table 8", "Appendix D", "[4c]"). The design is in
scripts/report_html_template.html.

    python3 scripts/report_html.py                      # writes docs/report.html
    python3 scripts/report_html.py --out page.html --link-base URL --fragment

By default the page is a complete standalone document and links to repository
files stay relative, so they work in a clone. --link-base rewrites them to a
URL (for example a GitHub blob URL), and --fragment leaves out the document
skeleton for hosts that supply their own. Needs the Python-Markdown package
(pip install markdown) in addition to the standard library.
"""
import argparse
import base64
import hashlib
import html
import os
import re

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
TEMPLATE = os.path.join(HERE, "report_html_template.html")
STANDALONE_RESET = "body { margin: 0; } img { max-width: 100%; } [hidden] { display: none !important; }"


def figures(md_text):
    """An image line followed by its 'Figure N.' caption becomes one <figure>."""
    def sub(m):
        alt, path, num, cap = m.groups()
        data = base64.b64encode(open(os.path.join(REPO, "docs", path), "rb").read()).decode()
        cap_html = markdown.markdown(cap.strip()).removeprefix("<p>").removesuffix("</p>")
        return (f'\n<figure id="fig-{num}"><div class="plate"><img src="data:image/svg+xml;base64,{data}" '
                f'alt="{html.escape(alt)}" loading="lazy"></div><figcaption><span class="cap-label">'
                f'Figure {num}.</span> {cap_html}</figcaption></figure>\n')
    return re.sub(r"!\[([^\]]*)\]\((figures/[^)]+\.svg)\)\s*\n\s*\n\*\*Figure (\d+)\.\*\*([^\n]*)", sub, md_text)


def mermaid(md_text):
    return re.sub(r"```mermaid\n(.*?)```",
                  lambda m: f'\n<div class="diagram"><pre class="mermaid">{html.escape(m.group(1).strip())}</pre></div>\n',
                  md_text, flags=re.S)


def rebase_links(md_text, base):
    """Rewrite relative repository links (relative to docs/) onto base."""
    def sub(m):
        label, target = m.groups()
        if target.startswith(("http", "#", "mailto:")):
            return m.group(0)
        path, _, frag = target.partition("#")
        path = os.path.normpath(os.path.join("docs", path)).replace("\\", "/")
        return f"[{label}]({base}{path}{'#' + frag if frag else ''})"
    return re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", sub, md_text)


def heading_ids(body):
    """Numbered sections get ids s-5-2, appendices app-a; returns the contents entries."""
    toc = []

    def sub(m):
        level, inner = int(m.group(1)), m.group(3)
        text = re.sub(r"<[^>]+>", "", inner)
        sec = re.match(r"(\d+(?:\.\d+)?)\.?\s", text)
        app = re.match(r"Appendix ([A-Z])\.", text)
        hid = ("s-" + sec.group(1).replace(".", "-")) if sec else ("app-" + app.group(1).lower()) if app else m.group(2)
        if level in (2, 3) and text != "Appendices":
            toc.append((level, hid, text))
        return f'<h{level} id="{hid}">{inner}</h{level}>'
    return re.sub(r'<h([1-6]) id="([^"]+)">(.*?)</h\1>', sub, body), toc


def captions_and_tables(body):
    body = re.sub(r"<p><strong>(Table ([0-9A-Z]+[0-9a-z]*))\.</strong>",
                  lambda m: f'<p class="caption" id="tab-{m.group(2).lower()}"><span class="cap-label">{m.group(1)}.</span>',
                  body)
    body = re.sub(r"<p><strong>(Diagram A)\.</strong>", r'<p class="caption" id="diagram-a"><span class="cap-label">\1.</span>', body)
    return body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")


def reference_ids(body):
    """Ids ref-N on the numbered sources (honouring each list's start) and ref-Na on the lettered ones."""
    def ol(m):
        start = re.match(r'<ol(?: start="(\d+)")?>', m.group(0))
        n = [int(start.group(1)) - 1 if start.group(1) else 0]

        def li(_):
            n[0] += 1
            return f'<li id="ref-{n[0]}">'
        return re.sub(r"<li>", li, m.group(0))
    cut = body.find('id="app-j"')
    if cut == -1:
        return body
    tail = re.sub(r'<ol(?: start="\d+")?>.*?</ol>', ol, body[cut:], flags=re.S)
    tail = re.sub(r"<p>\[(\d+[a-z])\]", lambda m: f'<p class="ref-sub" id="ref-{m.group(1)}">[{m.group(1)}]', tail)
    return body[:cut] + tail


def link_cross_references(body):
    """Link citations and Section/Figure/Table/Appendix references in running
    text, leaving headings, links, code and preformatted blocks alone."""
    def linkify(text):
        text = re.sub(r"\[(\d+[a-z]?)\](?!\()", lambda m: f'<a class="cite" href="#ref-{m.group(1)}">[{m.group(1)}]</a>', text)
        text = re.sub(r"\b(Sections?) (\d+(?:\.\d+)?)\b",
                      lambda m: f'{m.group(1)} <a href="#s-{m.group(2).replace(".", "-")}">{m.group(2)}</a>', text)
        text = re.sub(r"\b(Figures?) (\d+)\b", lambda m: f'{m.group(1)} <a href="#fig-{m.group(2)}">{m.group(2)}</a>', text)
        text = re.sub(r"\b(Tables?) ([0-9]+[a-z]?|[A-G][0-9]+[a-z]?)\b",
                      lambda m: f'{m.group(1)} <a href="#tab-{m.group(2).lower()}">{m.group(2)}</a>', text)
        return re.sub(r"\b(Appendix) ([A-J])\b", lambda m: f'{m.group(1)} <a href="#app-{m.group(2).lower()}">{m.group(2)}</a>', text)

    out, skip = [], 0
    for tok in re.split(r"(<[^>]+>)", body):
        if tok.startswith("<"):
            t = tok.lower()
            if re.match(r"<(a|code|pre|h[1-6])[\s>]", t):
                skip += 1
            elif re.match(r"</(a|code|pre|h[1-6])>", t):
                skip = max(0, skip - 1)
            out.append(tok)
        else:
            out.append(tok if skip else linkify(tok))
    return "".join(out)


def contents(toc):
    out = []
    for i, (level, hid, text) in enumerate(toc):
        a = f'<a href="#{hid}">{html.escape(text)}</a>'
        if level == 3:
            out.append(f'<li class="sub">{a}</li>')
            if i + 1 == len(toc) or toc[i + 1][0] == 2:
                out.append("</ol></li>")
        elif i + 1 < len(toc) and toc[i + 1][0] == 3:
            out.append(f"<li>{a}<ol>")
        else:
            out.append(f"<li>{a}</li>")
    return "\n".join(out)


def build(link_base, fragment):
    src = open(os.path.join(REPO, "docs", "REPORT.md"), encoding="utf-8").read()
    lines = src.split("\n")
    title, scope, byline = lines[0].lstrip("# ").strip(), lines[2].strip().strip("*"), lines[4].strip()
    md_text = mermaid(figures("\n".join(lines[6:])))
    if link_base:
        md_text = rebase_links(md_text, link_base)
    body = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists", "toc"]).convert(md_text)
    body, toc = heading_ids(body)
    body = link_cross_references(reference_ids(captions_and_tables(body)))
    ids = set(re.findall(r'id="([^"]+)"', body))
    dangling = sorted({h for h in re.findall(r'href="#([^"]+)"', body) if h not in ids})
    if dangling:
        raise SystemExit(f"cross-references without a target: {dangling}")
    page = (open(TEMPLATE, encoding="utf-8").read()
            .replace("{{TITLE}}", html.escape(title)).replace("{{SCOPE}}", html.escape(scope))
            .replace("{{BYLINE}}", html.escape(byline)).replace("{{TOC}}", contents(toc))
            .replace("{{BODY}}", body).replace("{{SOURCE_SHA}}", hashlib.sha256(src.encode()).hexdigest()[:12]))
    if not fragment:
        head, sep, rest = page.partition("</style>")
        head = head.replace('<meta name="claude:viewport-fit-sides" content="cover">\n', "")
        page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                f"{head}{STANDALONE_RESET}\n{sep}\n</head>\n<body>\n{rest.lstrip()}</body>\n</html>\n")
    return page, len(toc)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "report.html"))
    ap.add_argument("--link-base", default="", help="URL that relative repository links are rewritten onto")
    ap.add_argument("--fragment", action="store_true", help="leave out the document skeleton")
    a = ap.parse_args()
    page, n_toc = build(a.link_base, a.fragment)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {a.out}: {len(page) / 1024:.0f} KB, {n_toc} contents entries")


if __name__ == "__main__":
    main()
