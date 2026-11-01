"""The README's fairgoers shot: a group at the amusement park (examples/amusement_park), each
holding something from the stalls. Exports the example, opens the runtime in a window over the
control API, hands the biggest group a hot dog, a lemonade and a balloon, and frames them from
the front. Run from the repository root (docs/make_screenshots.sh does):

    .venv/bin/python docs/make_fair_shot.py docs/images/godot_fairgoers.png
"""

import math
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from geogen import roots
from geogen.export import export_scene
from geogen.registry import SceneRegistry
from geogen.runtime_client import RuntimeClient

SCENE = "amusement_park"
ITEMS = ["hot_dog", "lemonade_cup", "balloon"]


def main(out: Path) -> None:
    generated = Path(tempfile.mkdtemp(prefix="geogen_fair_")).resolve()
    roots.use(Path("examples/amusement_park"))
    try:
        registry = SceneRegistry()
        registry.discover()
        export_scene(registry[SCENE](), generated / f"{SCENE}.glb")
    finally:
        roots.use()
    rt = RuntimeClient.launch(SCENE, generated=str(generated), engine_args=())
    try:
        rt.call("wait_ready", timeout=120)
        groups = defaultdict(list)
        for npc in rt.call("npcs.list"):
            if npc.get("group"):
                groups[npc["group"]].append(npc)
        group = max(groups.values(), key=len)
        for npc, item in zip(group, ITEMS):
            rt.call("inventory.give", item=item, actor=npc["actor"])
        rt.call("time.advance", seconds=0.5)
        people = [n for n in rt.call("npcs.list") if n["npc"] in {g["npc"] for g in group}]
        centre = [sum(p["position"][i] for p in people) / len(people) for i in range(3)]
        yaw = math.radians(people[0]["yaw"])
        ahead = (math.sin(yaw), math.cos(yaw))
        at = [centre[0], centre[1] + 1.0, centre[2]]
        frm = [at[0] + ahead[0] * 3.6 + ahead[1] * 1.2, at[1] + 0.5, at[2] + ahead[1] * 3.6 - ahead[0] * 1.2]
        rt.call("capture.screenshot", path=str(out.resolve()), **{"from": frm, "at": at}, fov=45)
        print("fairgoers:", [(p["npc"], p.get("holding")) for p in people])
    finally:
        rt.close()


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "docs/images/godot_fairgoers.png"))
