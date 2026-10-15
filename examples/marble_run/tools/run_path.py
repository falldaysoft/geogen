"""Writes assets/ball_track.yaml: the marble run's trough (one sweep along a spline, so a
rolling ball meets no seams), the posts under it, the pole in the helix and its spokes.

The centreline is a ramp, a U-turn, a ramp back, two turns of a helix round a pole and a
last ramp out over the exit funnel, falling at a steady 4 degrees. The balls are physics
bodies the whole way (the runtime is Jolt, which rolls a sphere over a trimesh's internal
edges); their damping keeps them slow enough to stay in the trough round the bends.
Edit this file, not the YAML:

    .venv/bin/python examples/marble_run/tools/run_path.py
"""
import math
from pathlib import Path

GRADE = math.tan(math.radians(4.0))      # fall per metre along the run
STEP = 0.4                               # m between spline points
TOP = 2.9                                # y of the trough's floor at its closed back end
LEAD = 0.6                               # m of trough behind the drop point (the origin)
INNER, WALL = 0.08, 0.014                # the trough: inner radius (a 12 cm ball), wall
POLE = (0.5, 2.0)                        # the helix's centre (x, z) and radius
RADIUS = 0.8

points = []
y = TOP


def add(x, z):
    global y
    if points:
        px, _, pz = points[-1]
        y -= math.hypot(x - px, z - pz) * GRADE
    points.append((x, y, z))


def line(x0, z0, x1, z1):
    n = max(1, round(math.hypot(x1 - x0, z1 - z0) / STEP))
    for k in range(1, n + 1):
        add(x0 + (x1 - x0) * k / n, z0 + (z1 - z0) * k / n)


def arc(cx, cz, r, a0, a1):
    n = max(2, round(abs(a1 - a0) * r / STEP))
    for k in range(1, n + 1):
        a = a0 + (a1 - a0) * k / n
        add(cx + r * math.cos(a), cz + r * math.sin(a))


add(-LEAD, 0.0)
line(-LEAD, 0.0, 3.0, 0.0)                                 # ramp east, from behind the drop point
arc(3.0, 0.6, 0.6, -math.pi / 2, math.pi / 2)             # U-turn, back west at z = 1.2
line(3.0, 1.2, POLE[0], 1.2)                               # ramp west
helix_start = len(points) - 1
arc(POLE[0], POLE[1], RADIUS, -math.pi / 2, -math.pi / 2 - 4 * math.pi)   # two turns round the pole
helix_end = len(points) - 1
line(POLE[0], 1.2, -2.0, 1.2)                              # last ramp, over the catch bowl


def fmt(v):
    return "%.3f" % v


# The trough's cross-section: a half pipe, its floor on the path (x right, y up).
profile = []
for k in range(13):                                        # inner arc, left rim to right rim
    a = math.pi + math.pi * k / 12
    profile.append((INNER * math.cos(a), INNER + INNER * math.sin(a)))
for k in range(12, -1, -1):                                # outer arc back
    a = math.pi + math.pi * k / 12
    profile.append(((INNER + WALL) * math.cos(a), INNER + (INNER + WALL) * math.sin(a)))

parts = []
back = points[0]
parts.append(f"""  stop:
    primitive: cube
    size: [0.02, {fmt(INNER + WALL)}, {fmt(2 * (INNER + WALL))}]
    anchor: bottom_center
    offset: [{fmt(back[0] - 0.005)}, {fmt(back[1] - WALL)}, {fmt(back[2])}]
    bevel: 0
    material: brass""")
parts.append(f"""  trough:
    primitive: sweep
    profile: {{outer: [{", ".join(f"[{fmt(x)}, {fmt(y)}]" for x, y in profile)}]}}
    path: {{spline: [{", ".join("[" + ", ".join(fmt(v) for v in p) + "]" for p in points)}], samples: 4}}
    center: false
    level: true
    material: brass
    collider: mesh""")

# Posts under the straights and the U-turn, from the floor to the trough's underside.
posts = [points[k] for k in (2, 6, 10, 15, 18)] + [points[-6], points[-3]]
for n, (x, y0, z) in enumerate(posts):
    h = y0 - WALL - 0.002
    parts.append(f"""  post_{n}:
    primitive: cylinder
    size: [0.05, {fmt(h)}, 0.05]
    anchor: bottom_center
    offset: [{fmt(x)}, 0, {fmt(z)}]
    material: wood_dark""")

top = points[helix_start][1]
parts.append(f"""  pole:
    primitive: cylinder
    size: [0.1, {fmt(top + 0.25)}, 0.1]
    anchor: bottom_center
    offset: [{fmt(POLE[0])}, 0, {fmt(POLE[1])}]
    material: wood_dark
  cap:
    primitive: sphere
    size: [0.14, 0.14, 0.14]
    attach_to: pole
    at: top
    from: center
    material: brass""")

# Spokes from the pole to the trough's underside, four a turn.
for n, k in enumerate(range(helix_start + 1, helix_end + 1, 3)):
    x, y0, z = points[k]
    dx, dz = x - POLE[0], z - POLE[1]
    length = math.hypot(dx, dz) - 0.05 - INNER * 0.5
    yaw = math.degrees(math.atan2(dx, dz))
    mx = POLE[0] + dx / math.hypot(dx, dz) * (0.05 + length / 2)
    mz = POLE[1] + dz / math.hypot(dx, dz) * (0.05 + length / 2)
    parts.append(f"""  spoke_{n}:
    primitive: cube
    size: [0.03, 0.03, {fmt(length)}]
    anchor: bottom_center
    offset: [{fmt(mx)}, {fmt(y0 - WALL - 0.032)}, {fmt(mz)}]
    rotation: [0, {fmt(yaw)}, 0]
    bevel: 0
    material: brass""")

length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
drop_y = TOP - LEAD * GRADE

text = f"""# The marble run's track: a brass half-pipe trough that ramps east, U-turns, ramps back,
# winds twice round a pole and runs out west over the exit funnel ({length:.1f} m, falling
# {TOP - points[-1][1]:.2f} m at a steady 4 degrees), on wooden posts and brass spokes. Its
# floor is at y = {fmt(drop_y)} at the origin, under the hopper ({LEAD} m of trough and a
# stop behind it), and ends at
# [{fmt(points[-1][0])}, {fmt(points[-1][1])}, {fmt(points[-1][2])}], heading -X.
# The hopper drops balls onto the floor at the origin; they roll down it as physics bodies
# and fall off the end into the funnel.
# GENERATED by tools/run_path.py: edit that, not this.

name: ball_track
tags: [machine.marble_run]
origin: bottom_center
size: [1, 1, 1]
bounds: geometry

parts:
{chr(10).join(parts)}

attachments:
  start: {{anchor: bottom_center, offset: [0, {fmt(drop_y)}, 0], offset_mode: absolute}}
  end:   {{anchor: bottom_center, offset: [{fmt(points[-1][0])}, {fmt(points[-1][1])}, {fmt(points[-1][2])}], offset_mode: absolute}}
"""
out = Path(__file__).resolve().parent.parent / "assets" / "ball_track.yaml"
out.write_text(text)
print(f"wrote {out}: {len(points)} points, {length:.2f} m, {TOP - points[-1][1]:.2f} m fall")
