class_name GeogenHudCrosshair
extends Control
## The aim point: a small dot in the middle of the screen, and a ring round it while E
## would do something (an interaction, an NPC to greet, a seat). It has no logic of its
## own: main tells it what the focus is (`aim(active)`), as it tells the E prompt.

const DOT := 2.0
const RING := 9.0
const LIGHT := Color(1, 1, 1, 0.9)
const SHADE := Color(0, 0, 0, 0.55)
const ACTIVE := Color(1.0, 0.82, 0.4, 0.95)       # the tracker's gold

var active := false
var shown := true


func _init() -> void:
    name = "crosshair"
    mouse_filter = Control.MOUSE_FILTER_IGNORE
    set_anchors_preset(Control.PRESET_FULL_RECT)


## `on`: something usable is aimed at. `visible_`: the player's own view is on screen.
func aim(on: bool, visible_ := true) -> void:
    if on == active and visible_ == shown:
        return
    active = on
    shown = visible_
    visible = visible_
    queue_redraw()


func _draw() -> void:
    var centre := size / 2.0
    if active:
        draw_arc(centre, RING + 1.0, 0.0, TAU, 40, SHADE, 3.5, true)
        draw_arc(centre, RING, 0.0, TAU, 40, ACTIVE, 2.0, true)
    draw_circle(centre, DOT + 1.2, SHADE)
    draw_circle(centre, DOT, ACTIVE if active else LIGHT)


func hud_report() -> Dictionary:
    return {"visible": visible, "active": active}
