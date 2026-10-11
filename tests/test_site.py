"""The HTML catalogue (docs/make_site.py) built from the committed markdown pages."""

import importlib.util
import re
from pathlib import Path

import pytest

DOCS = Path(__file__).parent.parent / "docs"


@pytest.fixture(scope="module")
def make_site():
    spec = importlib.util.spec_from_file_location("make_site", DOCS / "make_site.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def site(make_site, tmp_path_factory):
    out = tmp_path_factory.mktemp("site")
    make_site.Site(DOCS / "catalogue", "https://github.com/example/geogen", "main").build(out)
    return out


def test_every_page_is_built_and_every_link_resolves(site):
    pages = sorted((DOCS / "catalogue").rglob("*.md"))
    built = sorted(site.rglob("*.html"))
    assert len(built) == len(pages) and (site / "index.html").exists() and (site / "assets" / "door.html").exists()
    for page in built:
        text = page.read_text()
        ids = set(re.findall(r'id="([^"]+)"', text))
        for url in re.findall(r'\b(?:href|src)="([^"]+)"', text):
            if url.startswith("#"):
                assert url[1:] in ids, f"{page.name}: no anchor {url}"
            elif not re.match(r"^[a-z]+:", url):
                assert (page.parent / url.split("#")[0]).resolve().exists(), f"{page.name}: broken link {url}"
        # No markdown left outside code.
        prose = re.sub(r"<pre>.*?</pre>|<code>.*?</code>", "", text, flags=re.S)
        assert not re.search(r"\*\*|\]\(|^\||^#+ ", prose, flags=re.M), page.name


def test_pages_link_home_and_to_the_source(site):
    index = (site / "index.html").read_text()
    assert index.count('class="card"') == len(list((site / "assets").glob("*.html")))
    assert 'class="filter"' in index and '<a class="card" href="assets/door.html" data-name="door">' in index
    door = (site / "assets" / "door.html").read_text()
    assert 'href="https://github.com/example/geogen/blob/main/assets/door.yaml"' in door
    assert 'href="../style.css"' in door and 'src="../img/door.png"' in door
    assert (site / "img" / "door.png").exists() and (site / ".nojekyll").exists()
    assert not (site / "img" / "hashes.json").exists()


def test_markdown_subset(make_site):
    site = make_site.Site(DOCS / "catalogue", "https://github.com/example/geogen", "main")
    title, body = site.body("# T\n\ntext with <room>.<surface> and `a|b` and **bold**\n\n"
                            "| Key | Meaning |\n| --- | --- |\n| `on` | use \\| enter [x](other.md#part) |\n\n"
                            "```yaml\nplace:  # things\n  a: {b: 1}\n```\n\n## Two words\n", "page.md")
    assert title == "T"
    assert "&lt;room&gt;.&lt;surface&gt;" in body and "<code>a|b</code>" in body and "<strong>bold</strong>" in body
    assert '<td>use | enter <a href="other.html#part">x</a></td>' in body
    assert '<span class="k">place</span>:  <span class="c"># things</span>' in body
    assert '<h2 id="two-words">' in body
    assert site.url("README.md", "assets/door.md") == "index.html"
    assert site.url("../README.md", "assets/door.md") == "../index.html"
