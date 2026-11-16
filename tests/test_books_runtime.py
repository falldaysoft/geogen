"""Books in the runtime (geogen.104): the loose books on the cottage's bookshelf are physics props
whose own `read` affordance NPCs take up: go to the book, pick it up, read it, put it back where it
was. Nothing in the code is about books; a carryable prop's affordance gets the prop as its subject
and where it lies as its place (npc.gd `_prop_affordance`, `_where_it_lies`)."""

import pytest

pytestmark = [pytest.mark.godot, pytest.mark.xdist_group("test_books_runtime")]

BOOKS = [f"book_{i}" for i in range(1, 7)]


@pytest.fixture(scope="module")
def cottage(scene_export, godot_runtime):
    rt = godot_runtime("cottage", scene_export("cottage"))
    rt.call("wait_ready")
    rt.call("time.pause")
    return rt


def ana(rt):
    return next(n for n in rt.call("npcs.list") if n["npc"].startswith("ana"))


def book_state(rt, name="book_1"):
    found = rt.call("interactions.list")
    found = found.get("interactions", found) if isinstance(found, dict) else found
    return next(i["state"] for i in found if name in str(i.get("asset", i.get("node", ""))) or name in str(i.get("key", "")))


def book(rt, name):
    return next(b for b in rt.call("world.bodies") if b["node"] == name)


def test_the_books_are_offered_to_read(cottage):
    options = cottage.call("npcs.options", npc="ana_npc")
    options = options.get("options", options) if isinstance(options, dict) else options
    reads = {o["id"].split("#")[0] for o in options if o["action"] == "read"}
    assert reads == set(BOOKS)


def test_a_book_is_taken_down_read_and_put_back(cottage):
    rt = cottage
    before = book(rt, "book_1")
    curious = ana(rt)["needs"]["curiosity"]
    rt.call("npcs.force", npc="ana_npc", option="book_1#0")
    held, spent = None, 0.0
    while spent < 200 and held is None:
        rt.call("time.advance", seconds=2, scale=4)
        spent += 2
        if book(rt, "book_1")["holder"]:
            held = ana(rt)
    assert held is not None, f"never picked up: {ana(rt)}"
    assert held["prop"].endswith("book_1") and held["doing"].startswith("book_1#0")
    # It takes the book to a seat or a bed and opens it there.
    opened = None
    while spent < 300 and opened is None and book(rt, "book_1")["holder"]:
        rt.call("time.advance", seconds=2, scale=4)
        spent += 2
        if book_state(rt) == "open":
            opened = ana(rt)
    assert opened is not None, "the book was opened to read"
    assert opened["pose"] in ("sit", "lie"), f"read {opened['pose']}"
    while spent < 300 and book(rt, "book_1")["holder"]:
        rt.call("time.advance", seconds=2, scale=4)
        spent += 2
    rt.call("time.advance", seconds=2, scale=4)       # let it settle
    after = book(rt, "book_1")
    assert after["holder"] == ""
    moved = sum((a - b) ** 2 for a, b in zip(after["center"], before["center"])) ** 0.5
    assert moved < 0.03, f"put back {moved * 100:.1f} cm from where it was"
    assert abs((after["yaw_deg"] - before["yaw_deg"] + 180) % 360 - 180) < 5, "facing out as it was"
    assert ana(rt)["needs"]["curiosity"] > curious + 0.1, "reading fed her curiosity"
    assert book_state(rt) == "closed", "shut before it went back"
