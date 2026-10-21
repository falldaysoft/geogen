"""The HTML catalogue (docs/make_site.py) built from the committed markdown pages."""

import importlib.util
import json
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


def test_play_is_in_the_nav_only_with_a_web_build(make_site, tmp_path):
    # The web build of the Godot runtime (pages.yml copies it to play/ before making the pages).
    (tmp_path / "play").mkdir()
    (tmp_path / "play" / "index.html").write_text("<html></html>")
    make_site.Site(DOCS / "catalogue", "https://github.com/example/geogen", "main").build(tmp_path)
    assert '<a href="../play/">Play</a>' in (tmp_path / "assets" / "door.html").read_text()


def test_models_give_asset_pages_a_3d_view(make_site, tmp_path):
    """With models/ (docs/make_models.py), the viewer goes in at view/ and listed assets' pictures open it."""
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "index.json").write_text('{"assets": ["door"]}')
    make_site.Site(DOCS / "catalogue", "https://github.com/example/geogen", "main").build(tmp_path)
    assert (tmp_path / "view" / "index.html").exists() and (tmp_path / "view" / "viewer.js").exists()
    door = (tmp_path / "assets" / "door.html").read_text()
    assert '<button class="view3d" data-src="../view/?m=door"' in door and 'href="../view/?m=door"' in door
    assert 'href="../view/"' in door                       # the nav
    assert "view3d" not in (tmp_path / "assets" / "sofa.html").read_text()


def test_web_builds_are_listed_on_the_index(make_site, tmp_path):
    """Builds at play/ and play/<project>/ (each with a build.json and a page) get a line on the index."""
    out = tmp_path / "site"
    for where, info in (("play", {"name": "geogen", "description": "The showcases"}),
                        ("play/sandwich", {"name": "sandwich", "description": "A kitchen"}),
                        ("play/half_built", {"name": "no page", "description": "left out"})):
        (out / where).mkdir(parents=True)
        (out / where / "build.json").write_text(json.dumps(info))
        if where != "play/half_built":
            (out / where / "index.html").write_text("<html></html>")
    make_site.Site(DOCS / "catalogue", "https://github.com/example/geogen", "main").build(out)
    index = (out / "index.html").read_text()
    assert '<a href="play/">geogen</a> · The showcases' in index
    assert '<a href="play/sandwich/">sandwich</a> · A kitchen' in index
    assert "half_built" not in index and 'href="play/"' in index    # the nav's Play link too


def test_an_exported_page_can_load_a_shared_engine():
    """export_web.point_at_engine: the engine from engine/, the pack from beside the page."""
    spec = importlib.util.spec_from_file_location(
        "export_web", DOCS.parent / "runtime" / "godot" / "tools" / "export_web.py")
    export_web = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_web)
    page = ('<script src="index.js"></script>\n<script>\nconst GODOT_CONFIG = {"args":[],"executable":"index",'
            '"fileSizes":{"index.pck":10,"index.wasm":20}};\nconst GODOT_THREADS_ENABLED = false;\n</script>')
    out = export_web.point_at_engine(page, "../engine")
    config = json.loads(re.search(r"const GODOT_CONFIG = (\{.*\});", out).group(1))
    assert config["executable"] == "../engine/index" and config["mainPack"] == "index.pck"
    assert config["fileSizes"] == {"index.pck": 10, "../engine/index.wasm": 20}
    assert '<script src="../engine/index.js"></script>' in out and "GODOT_THREADS_ENABLED = false" in out
