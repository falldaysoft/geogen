"""Write the motion conformance traces (tests/data/motion/*.json) from the review scenarios the
runtime rig can run: python tests/data/motion/make_traces.py

Each is geogen.motion_review.motion_trace(scenario): the body's rig extras, the motions, the
script, and the reference rig's bone positions and events. runtime/godot/scripts/motion_rig.gd
replays them (tests/test_motion_godot.py)."""

import json
from pathlib import Path

from geogen.motion_review import motion_trace

SCENARIOS = ["squat", "jumping_jacks", "squat_shoved", "bow_wave", "emotes", "idles", "sit_on_chair", "sit_watch_tv",
             "watch_tv", "watch_by_turns", "drink", "inspect", "take_carry_set_down", "bite_apple", "bite_apple_left"]

if __name__ == "__main__":
    here = Path(__file__).parent
    for name in SCENARIOS:
        trace = motion_trace(name)
        path = here / f"{name}.json"
        path.write_text(json.dumps(trace, separators=(",", ":")) + "\n")
        print(f"{path}: {trace['frames']} frames, {len(trace['expect']['positions'])} checked, "
              f"{len(trace['expect']['events'])} events, {path.stat().st_size // 1024} KB")
