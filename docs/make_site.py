#!/usr/bin/env python3
"""Build the browsable HTML catalogue (for GitHub Pages) from docs/catalogue/*.md.

    python docs/make_site.py [OUT_DIR]            # default: _site

The markdown pages are the source of truth (``docs/make_catalogue.sh`` writes
them from ``--describe``); this turns them into a static site: one HTML page
per markdown page, the thumbnails, and a stylesheet. It only needs the
standard library, so the Pages workflow (.github/workflows/pages.yml) runs it
on a bare checkout. Links that leave the catalogue (an asset's YAML source)
point at the repository on GitHub (``--repo``, ``--ref``).

It reads the subset of markdown the catalogue pages use: headings, paragraphs,
fenced code, tables, ``-`` lists, links, bold, code spans and the raw
``<img>`` / thumbnail ``<table>`` lines.
"""

from __future__ import annotations

import argparse
import html
import os
import posixpath
import re
import shutil
import subprocess
from pathlib import Path

CATALOGUE = Path(__file__).parent / "catalogue"
REPO_ROOT = Path(__file__).parent.parent
NAV = [("index.html", "Assets"), ("primitives.html", "Primitives"), ("vocabulary.html", "Vocabulary"),
       ("scenes.html", "Scenes")]
PLAY = ("play/", "Play")     # the web build of the Godot runtime, when the site has one (pages.yml)

CSS = """
:root {
  --bg: #f6f5f1; --panel: #ffffff; --ink: #1d2126; --muted: #5d6670; --line: #dcd9d0;
  --accent: #b4532a; --code-bg: #efede6; --key: #2f5f8a; --comment: #7a8189; --shadow: 0 1px 2px rgba(0,0,0,.06);
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #16181b; --panel: #1e2125; --ink: #e4e2dc; --muted: #9aa1a9; --line: #30343a;
          --accent: #e58a5e; --code-bg: #24282d; --key: #8bb8e0; --comment: #7f8791; --shadow: none; }
}
* { box-sizing: border-box; }
html { scroll-padding-top: 4.5rem; }
body { margin: 0; background: var(--bg); color: var(--ink);
       font: 16px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
header.site { position: sticky; top: 0; z-index: 5; background: var(--panel); border-bottom: 1px solid var(--line); }
header.site .bar { max-width: 72rem; margin: 0 auto; padding: .6rem 1rem; display: flex; flex-wrap: wrap;
                   align-items: center; gap: .4rem 1.4rem; }
header.site .brand { font-weight: 700; color: var(--ink); letter-spacing: .01em; }
header.site nav { display: flex; flex-wrap: wrap; gap: .2rem 1rem; }
header.site nav a { color: var(--muted); padding: .15rem 0; border-bottom: 2px solid transparent; }
header.site nav a.here { color: var(--ink); border-color: var(--accent); }
header.site .repo { margin-left: auto; color: var(--muted); font-size: .9rem; }
main { max-width: 72rem; margin: 0 auto; padding: 1.2rem 1rem 4rem; }
h1 { font-size: 1.9rem; margin: .6rem 0 .8rem; text-wrap: balance; }
h2 { font-size: 1.25rem; margin: 2.2rem 0 .7rem; padding-bottom: .3rem; border-bottom: 1px solid var(--line); }
h3 { font-size: 1.05rem; margin: 1.6rem 0 .5rem; }
h2 .anchor, h3 .anchor { margin-left: .4rem; color: var(--line); font-weight: 400; }
h2:hover .anchor, h3:hover .anchor { color: var(--accent); }
p, ul { margin: .7rem 0; max-width: 60rem; }
code { font: .88em/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
       background: var(--code-bg); padding: .08em .32em; border-radius: 4px; }
pre { background: var(--code-bg); border: 1px solid var(--line); border-radius: 8px; padding: .8rem 1rem;
      overflow-x: auto; max-width: 60rem; }
pre code { background: none; padding: 0; font-size: .84rem; }
pre .k { color: var(--key); } pre .c { color: var(--comment); font-style: italic; }
.table-wrap { overflow-x: auto; margin: .8rem 0; border: 1px solid var(--line); border-radius: 8px;
              background: var(--panel); box-shadow: var(--shadow); }
table { border-collapse: collapse; width: 100%; font-size: .92rem; }
th, td { text-align: left; vertical-align: top; padding: .42rem .7rem; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; font-size: .8rem; text-transform: uppercase; letter-spacing: .04em;
     white-space: nowrap; }
tr:last-child td { border-bottom: none; }
td:first-child { white-space: nowrap; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(9.5rem, 1fr)); gap: .8rem; margin: .8rem 0; }
.card { display: flex; flex-direction: column; background: var(--panel); border: 1px solid var(--line);
        border-radius: 10px; overflow: hidden; color: var(--ink); box-shadow: var(--shadow);
        transition: transform .12s, border-color .12s; }
.card:hover { transform: translateY(-2px); border-color: var(--accent); text-decoration: none; }
.card img { width: 100%; aspect-ratio: 1; object-fit: cover; display: block; background: var(--code-bg); }
.card span { padding: .4rem .6rem .5rem; font: .8rem/1.3 ui-monospace, SFMono-Regular, Menlo, monospace;
             overflow-wrap: anywhere; }
.shots { display: flex; flex-wrap: wrap; gap: .8rem; max-width: none; }
.shots img { height: auto; max-width: 100%; border-radius: 10px; border: 1px solid var(--line); }
.filter { width: 100%; max-width: 26rem; margin: .4rem 0 0; padding: .55rem .8rem; font: inherit; color: var(--ink);
          background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
.filter:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
.empty { color: var(--muted); }
.crumb { font-size: .9rem; color: var(--muted); margin: 0; }
footer { max-width: 72rem; margin: 0 auto; padding: 0 1rem 2rem; color: var(--muted); font-size: .85rem; }
@media (prefers-reduced-motion: reduce) { .card { transition: none; } .card:hover { transform: none; } }
"""

FILTER_JS = """
const box = document.querySelector('.filter');
const cards = [...document.querySelectorAll('.card')];
const empty = document.querySelector('.empty');
function apply() {
  const q = box.value.trim().toLowerCase();
  let shown = 0;
  for (const c of cards) { const ok = c.dataset.name.includes(q); c.hidden = !ok; shown += ok; }
  for (const g of document.querySelectorAll('.grid')) {
    const any = [...g.children].some(c => !c.hidden);
    g.hidden = !any;
    if (g.previousElementSibling && g.previousElementSibling.tagName === 'H2') g.previousElementSibling.hidden = !any;
  }
  empty.hidden = shown > 0;
}
box.addEventListener('input', apply);
document.addEventListener('keydown', e => {
  if (e.key === '/' && document.activeElement !== box) { e.preventDefault(); box.focus(); }
});
"""

_CARD = re.compile(r'<td[^>]*><a href="([^"]+)"><img src="([^"]+)"[^>]*alt="([^"]*)"[^>]*><br>([^<]*)</a></td>')
_ATTR_URL = re.compile(r'\b(href|src)="([^"]+)"')
_INLINE = re.compile(r"(`[^`]+`)|\[([^\]]+)\]\(([^)\s]+)\)|\*\*([^*]+)\*\*")


def slug(text: str) -> str:
    """GitHub's heading anchor for ``text`` (the pages link to each other's headings with these)."""
    text = re.sub(r"`|\*\*", "", text).strip().lower()
    return re.sub(r"\s", "-", re.sub(r"[^\w\- ]", "", text))


class Site:
    def __init__(self, source: Path, repo: str, ref: str) -> None:
        self.source = source.resolve()
        self.repo = repo.rstrip("/")
        self.ref = ref
        self.play = False

    # -- links -----------------------------------------------------------------------------

    def url(self, target: str, page: str) -> str:
        """``target`` (as written in the markdown page ``page``) for the HTML site."""
        if re.match(r"^[a-z]+:|^#", target):
            return target
        path, hash_, fragment = target.partition("#")
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(page), path))
        if resolved.startswith(".."):       # outside the catalogue: the file in the repository
            in_repo = os.path.relpath(self.source / posixpath.dirname(page) / path, REPO_ROOT)
            return f"{self.repo}/blob/{self.ref}/{Path(in_repo).as_posix()}{hash_}{fragment}"
        if path.endswith(".md"):
            head, _, name = path.rpartition("/")
            name = "index.html" if name == "README.md" else name[:-3] + ".html"
            path = f"{head}/{name}" if head else name
        return f"{path}{hash_}{fragment}"

    # -- inline ----------------------------------------------------------------------------

    def inline(self, text: str, page: str) -> str:
        text = text.replace("\\|", "|")
        out, last = [], 0
        for match in _INLINE.finditer(text):
            out.append(html.escape(text[last:match.start()], quote=False))
            code, label, target, bold = match.groups()
            if code:
                out.append(f"<code>{html.escape(code[1:-1], quote=False)}</code>")
            elif label:
                out.append(f'<a href="{html.escape(self.url(target, page))}">{self.inline(label, page)}</a>')
            else:
                out.append(f"<strong>{html.escape(bold, quote=False)}</strong>")
            last = match.end()
        out.append(html.escape(text[last:], quote=False))
        return "".join(out)

    # -- blocks ----------------------------------------------------------------------------

    def raw(self, lines: list[str], page: str) -> str:
        """Raw HTML lines: thumbnail tables become a card grid, images a row of shots."""
        block = "\n".join(lines)
        block = _ATTR_URL.sub(lambda m: f'{m.group(1)}="{html.escape(self.url(m.group(2), page))}"', block)
        cards = _CARD.findall(block)
        if cards:
            return '<div class="grid">\n' + "\n".join(
                f'<a class="card" href="{href}" data-name="{html.escape(name.lower())}">'
                f'<img src="{src}" alt="{html.escape(alt)}" loading="lazy" width="256" height="256">'
                f"<span>{html.escape(name)}</span></a>" for href, src, alt, name in cards) + "\n</div>"
        if block.startswith("<img"):
            return f'<p class="shots">{block}</p>'
        return block

    def table(self, rows: list[str], page: str) -> str:
        def cells(row: str) -> list[str]:
            return [c.strip() for c in re.split(r"(?<!\\)\|", row.strip().strip("|"))]

        head, body = cells(rows[0]), [cells(r) for r in rows[2:]]
        out = ['<div class="table-wrap"><table>', "<thead><tr>"
               + "".join(f"<th>{self.inline(c, page)}</th>" for c in head) + "</tr></thead>", "<tbody>"]
        for row in body:
            out.append("<tr>" + "".join(f"<td>{self.inline(c, page)}</td>" for c in row) + "</tr>")
        return "\n".join([*out, "</tbody>", "</table></div>"])

    @staticmethod
    def code(lines: list[str], lang: str) -> str:
        def shade(line: str) -> str:
            text, comment = line, ""
            match = re.search(r"(^|\s)#.*$", line)
            if lang == "yaml" and match:
                text, comment = line[:match.start() + len(match.group(1))], line[match.start() + len(match.group(1)):]
            text = html.escape(text, quote=False)
            if lang == "yaml":
                text = re.sub(r"^(\s*(?:- )?)([\w.\-<>]+)(:)(?=\s|$)", r'\1<span class="k">\2</span>\3', text)
            return text + (f'<span class="c">{html.escape(comment, quote=False)}</span>' if comment else "")

        return "<pre><code>" + "\n".join(shade(line) for line in lines) + "</code></pre>"

    def body(self, text: str, page: str) -> tuple[str, str]:
        """(title, HTML) for one markdown page."""
        lines = text.splitlines()
        out: list[str] = []
        title, i = "", 0
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                i += 1
            elif line.startswith("```"):
                end = next(j for j in range(i + 1, len(lines) + 1) if j == len(lines) or lines[j].startswith("```"))
                out.append(self.code(lines[i + 1:end], line[3:].strip()))
                i = end + 1
            elif line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                label = line[level:].strip()
                if level == 1:
                    title = title or label
                    out.append(f"<h1>{self.inline(label, page)}</h1>")
                else:
                    anchor = slug(label)
                    out.append(f'<h{level} id="{anchor}">{self.inline(label, page)}'
                               f'<a class="anchor" href="#{anchor}" aria-label="Link to this section">#</a></h{level}>')
                i += 1
            elif line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|\-]+\|\s*$", lines[i + 1]):
                end = next((j for j in range(i, len(lines)) if not lines[j].startswith("|")), len(lines))
                out.append(self.table(lines[i:end], page))
                i = end
            elif line.startswith("<"):
                end = next((j for j in range(i, len(lines)) if not lines[j].startswith("<")), len(lines))
                out.append(self.raw(lines[i:end], page))
                i = end
            elif line.startswith("- "):
                end = next((j for j in range(i, len(lines)) if not lines[j].startswith("- ")), len(lines))
                out.append("<ul>\n" + "\n".join(f"<li>{self.inline(item[2:], page)}</li>" for item in lines[i:end])
                           + "\n</ul>")
                i = end
            else:
                end = next((j for j in range(i, len(lines))
                            if not lines[j].strip() or lines[j].startswith(("#", "```", "|", "<", "- "))), len(lines))
                end = max(end, i + 1)
                out.append(f"<p>{self.inline(' '.join(s.strip() for s in lines[i:end]), page)}</p>")
                i = end
        return title, "\n".join(out)

    # -- pages -----------------------------------------------------------------------------

    def page(self, text: str, page: str) -> str:
        title, body = self.body(text, page)
        root = "../" * page.count("/")
        here = "index.html" if page.startswith("assets/") or page == "README.md" else page[:-3] + ".html"
        nav = "".join(f'<a href="{root}{href}"{" class=here" if href == here else ""}>{label}</a>'
                      for href, label in NAV + ([PLAY] if self.play else []))
        extra = ""
        if page == "README.md":
            # The filter goes above the first category; the usage example stays with the intro.
            first = body.find("<h2")
            body = (body[:first] + '<input class="filter" type="search" placeholder="Filter assets (press /)" '
                    'aria-label="Filter assets by name">\n<p class="empty" hidden>No asset matches.</p>\n' + body[first:])
            extra = f"<script>{FILTER_JS}</script>"
        elif page.startswith("assets/"):
            body = f'<p class="crumb"><a href="{root}index.html">Assets</a> /</p>\n' + body
        name = "Geogen catalogue"
        full_title = name if title == name else f"{title} · {name}"
        return (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                f"<title>{html.escape(full_title)}</title>\n"
                f'<link rel="stylesheet" href="{root}style.css">\n</head>\n<body>\n'
                f'<header class="site"><div class="bar"><a class="brand" href="{root}index.html">Geogen catalogue</a>'
                f'<nav>{nav}</nav><a class="repo" href="{self.repo}">GitHub</a></div></header>\n'
                f"<main>\n{body}\n</main>\n"
                f"<footer>Generated from <code>docs/catalogue</code> by <code>docs/make_site.py</code>.</footer>\n"
                f"{extra}\n</body>\n</html>\n")

    def build(self, out: Path) -> int:
        out.mkdir(parents=True, exist_ok=True)
        self.play = (out / PLAY[0] / "index.html").exists()      # copied in before the pages are made
        count = 0
        for md in sorted(self.source.rglob("*.md")):
            page = md.relative_to(self.source).as_posix()
            target = out / ("index.html" if page == "README.md" else page[:-3] + ".html")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self.page(md.read_text(), page))
            count += 1
        (out / "style.css").write_text(CSS.lstrip())
        (out / ".nojekyll").write_text("")
        if (self.source / "img").is_dir():
            shutil.copytree(self.source / "img", out / "img", dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("*.json"))
        return count


def default_repo() -> str:
    """The repository's GitHub URL: $GITHUB_REPOSITORY in CI, else the origin remote."""
    if os.environ.get("GITHUB_REPOSITORY"):
        return f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{os.environ['GITHUB_REPOSITORY']}"
    try:
        remote = subprocess.run(["git", "-C", str(REPO_ROOT), "remote", "get-url", "origin"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "https://github.com/falldaysoft/geogen"
    match = re.search(r"github\.com[:/](.+?)(?:\.git)?$", remote)
    return f"https://github.com/{match.group(1)}" if match else "https://github.com/falldaysoft/geogen"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out", nargs="?", default="_site", help="output directory (default: _site)")
    parser.add_argument("--source", default=str(CATALOGUE), help="the markdown catalogue (default: docs/catalogue)")
    parser.add_argument("--repo", default=None, help="repository URL for source links (default: the origin remote)")
    parser.add_argument("--ref", default=os.environ.get("GITHUB_REF_NAME", "main"), help="branch source links point at")
    args = parser.parse_args()
    count = Site(Path(args.source), args.repo or default_repo(), args.ref).build(Path(args.out))
    print(f"wrote {count} pages to {args.out}")


if __name__ == "__main__":
    main()
