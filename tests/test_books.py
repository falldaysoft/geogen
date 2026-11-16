"""Books as objects (geogen.104): book.yaml is a carryable physics prop with a `read` affordance,
the bookshelf has a surface per shelf, and the cottage's loose books stand on them."""

import numpy as np

from geogen import roots
from geogen.layout.loader import LayoutLoader
from geogen.npc import load_actions


def test_a_book_is_a_prop_to_read():
    book = LayoutLoader().load(roots.find("book.yaml", roots.CORE))
    assert book.meta["physics"]["carry"] is True
    (read,) = book.meta["affordances"]
    assert read["action"] == "read" and read["advertises"] == {"curiosity": 0.5}
    assert book.interactions[0].name == "book" and set(book.interactions[0].states) == {"closed", "open"}
    steps = load_actions()["read"]["steps"]
    verbs = [next(k for k in s if k not in ("look", "at", "prefer")) for s in steps]
    assert verbs == ["go_to", "pick_up", "find", "go_to", "face", "pose", "use", "wait", "use", "pose", "go_to", "put_down"]


def test_the_bookshelf_has_a_surface_per_shelf():
    shelf = LayoutLoader().load(roots.find("bookshelf.yaml", roots.CORE))
    assert {"top", "shelf_0", "shelf_1", "shelf_2", "shelf_3"} <= set(shelf.list_surfaces())


def test_the_cottage_books_stand_on_its_shelves(built_scene):
    cottage = built_scene("cottage")
    books = [n for n in cottage.iter_nodes() if n.name.startswith("book_")]
    assert len(books) == 6
    shelf = next(n for n in cottage.iter_nodes() if n.name == "bookshelf")
    shelf_y = sorted({round(float(shelf.world_transform()[1, 3]) + y, 3) for y in (0.53, 0.98)})
    for b in books:
        y = float(b.world_transform()[1, 3])
        assert min(abs(y - s) for s in shelf_y) < 0.03, (b.name, y, shelf_y)
        assert abs(np.degrees(np.arctan2(b.world_transform()[0, 2], b.world_transform()[2, 2]))) < 1, "spine out"
