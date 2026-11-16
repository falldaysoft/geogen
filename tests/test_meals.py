"""What is carried changes with what is done with it (geogen.105): a meal's body has states for
how much is left, an NPC's `eat` sets it out on the table and eats it down, and the household's
items show in the hand while they are carried."""

import pytest

from geogen import roots
from geogen.flows import load_item
from geogen.layout.loader import LayoutLoader
from geogen.npc import _check_step, load_actions


def test_the_meal_has_states_for_what_is_left():
    plate = LayoutLoader().load(roots.find("meal_plate.yaml", roots.CORE))
    (food,) = plate.interactions
    assert food.name == "food" and list(food.states) == ["full", "half", "empty"] and food.initial == "full"


def test_eating_sets_the_meal_out_and_eats_it_down():
    steps = load_actions()["eat"]["steps"]
    assert {"set_out": "meal"} in steps
    assert [s["show"] for s in steps if "show" in s] == [{"meal": "half"}, {"meal": "empty"}]
    chair = LayoutLoader().load(roots.find("chair.yaml", roots.CORE), params={"at_table": 1})
    (seat,) = chair.meta["affordances"]
    assert seat["uses"]["eat"]["action"] == "eat"


@pytest.mark.parametrize("item", ["raw_food", "dirty_plate"])
def test_household_items_show_in_the_hand(item):
    assert "held" in load_item(roots.find(f"items/{item}.yaml", roots.CORE))


@pytest.mark.parametrize("bad", [{"set_out": "not an id"}, {"show": "half"}, {"show": {}}])
def test_set_out_and_show_are_checked(bad):
    with pytest.raises(ValueError):
        _check_step(bad, "test")
