"""Content packages in the Godot runtime (rules.gd): a deployed package's rules join the program
of each scene it names, and its state outlives the scene that is loaded."""

import json
import time

import pytest

from geogen.runtime_client import ApiError
from test_content import deploy, write_assets

pytestmark = pytest.mark.xdist_group("test_content_runtime")

PULLED, GATE, LAMP = "world:errand.lever_pulled", "annex:gate#gate", "annex:lamp#lit"


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    assets = write_assets(tmp_path_factory.mktemp("content_rt_assets"))
    out = tmp_path_factory.mktemp("generated_content_rt")
    deploy(assets, out)
    return out


@pytest.fixture(scope="module")
def rt(generated, godot_runtime):
    client = godot_runtime("camp", generated)
    client.call("wait_ready")
    return client


def value(rt, key, want=None, timeout=10.0):
    """A variable's value, once it is ``want`` (rules run with the frame's events)."""
    end = time.time() + timeout
    while True:
        got = rt.call("state.get", key=key)["value"]
        if want is None or got == want or time.time() > end:
            return got
        time.sleep(0.05)


def errand(rt):
    (quest,) = [q for q in rt.call("quests.list") if q["id"] == "errand"]
    return quest["package"], quest["status"], [(o["id"], o["state"]) for o in quest["objectives"]]


def test_a_package_wires_one_scene_to_state_read_in_another(rt):
    # at camp: the package's quest started with the game, and its rule hears the camp's lever
    assert errand(rt) == ("errand", "active", [("pull", "open"), ("plate", "waiting")])
    assert value(rt, PULLED) is False
    with pytest.raises(ApiError, match="no state"):                 # the annex's gate isn't here
        rt.call("state.get", key=GATE)
    assert rt.call("player.use", asset="lever")["used"][0]["ok"]
    assert value(rt, PULLED, True) is True
    assert value(rt, "actor:errand.pulls") == 1
    assert errand(rt)[2] == [("pull", "done"), ("plate", "open")]

    # in the annex, which has no rules of its own: what the camp's lever set opens the gate
    rt.call("world.switch", scene="annex")
    assert value(rt, GATE, "open") == "open"
    assert value(rt, LAMP, True) is True                            # a binding of the package's
    assert value(rt, PULLED) is True and value(rt, "actor:errand.pulls") == 1
    with pytest.raises(ApiError, match="no state"):
        rt.call("state.get", key="camp:lever#pull")
    assert errand(rt) == ("errand", "active", [("pull", "done"), ("plate", "open")])
    rt.call("player.teleport", pos=[-3, 0.1, 0], yaw=0)             # onto the plate: the second objective
    assert value(rt, "actor:errand.quest_errand", "complete") == "complete"

    # back at camp it is all still so, and the quest is listed once
    rt.call("world.switch", scene="camp")
    assert value(rt, PULLED) is True
    assert errand(rt) == ("errand", "complete", [("pull", "done"), ("plate", "done")])
    stored = rt.call("state.list")
    assert stored["world"][PULLED] is True and stored["actors"]["player"]["actor:errand.pulls"] == 1

    # a scene's things start over when it loads again, so the gate's rule is judged afresh
    rt.call("world.switch", scene="annex")
    assert value(rt, GATE, "open") == "open" and value(rt, LAMP, True) is True
    # and state changed here is what the camp reads next
    rt.call("state.set", key=PULLED, value=False)
    assert value(rt, LAMP, False) is False
    rt.call("world.switch", scene="camp")
    assert value(rt, PULLED) is False


def _json_lines(out: str, prefix: str) -> list[dict]:
    return [json.loads(line.removeprefix(prefix)) for line in out.splitlines() if line.startswith(prefix)]


def test_a_package_is_loaded_across_a_switch(run_godot, generated):
    """One headless run: state set at camp, then --switch to the annex, where a rule reads it."""
    out = run_godot(f"--generated={generated}", "--scene=camp", f"--set={PULLED}=true", "--events=changed",
                    "--switch=annex@0.5", "--switch=camp@1.5")
    assert [s["scene"] for s in _json_lines(out, "switched: ")] == ["annex", "camp"]
    assert "geogen: content package errand runs in camp" in out
    assert "geogen: content package errand runs in annex" in out
    assert out.index("content package errand runs in camp") < out.index("content package errand runs in annex")
    changed = {e["key"]: e for e in _json_lines(out, "event: ")}
    assert changed[PULLED]["data"]["to"] is True and changed[PULLED]["package"] == "camp"
    assert changed[GATE]["data"]["to"] == "open" and changed[GATE]["origin"] == "rules"
    assert changed[GATE]["package"] == "annex"
    assert "geogen rules:" not in out and "SCRIPT ERROR" not in out


def test_scenes_without_the_package_are_untouched(run_godot, generated, tmp_path):
    """A catalogue that deploys nothing: the annex has no state of the package's."""
    import shutil

    plain = tmp_path / "generated"
    shutil.copytree(generated, plain)
    index = json.loads((plain / "catalogue.json").read_text())
    index["packages"] = []
    (plain / "catalogue.json").write_text(json.dumps(index))
    out = run_godot(f"--generated={plain}", "--scene=annex", f"--set={PULLED}=true", "--events=changed", "--quit-after=20")
    assert "content package" not in out and _json_lines(out, "event: ") == []
