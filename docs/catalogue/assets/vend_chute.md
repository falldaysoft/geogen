# vend_chute

`asset: vend_chute.yaml` · [source](../../../assets/vend_chute.yaml) · `machine.vend_chute`

<img src="../img/vend_chute.png" width="256" alt="vend_chute"> <img src="../img/vend_chute_views.png" width="512" alt="vend_chute: front and side">

A vending machine's drop point. For one product of vending_machine.yaml: a source of
spawned items (geogen.flows) that makes nothing by itself (stopped, rate 0) and one
`item` each time it takes `emit`, so every press of a button gives another, up to
40 of them in the world. Items appear at its origin and are pushed forward
(+Z); a small flap above it, out of sight under the hood, is all there is to see.

```yaml
place:
  vend_chute: {asset: vend_chute.yaml, params: {item: items/cola.yaml}}
```

## Params

| Param | Default | Choices | Notes |
| --- | --- | --- | --- |
| `item` | `items/cola.yaml` | text |  |

## Size

0.05 x 0.05 x 0.05 m (x, y, z); geometry (-0.025, 0.13, -0.025) to (0.025, 0.134, 0.025) from the origin; 12 triangles

## Parts

`flap` (cube)

## Attachments

| Attachment | Position | Facing |
| --- | --- | --- |
| `mouth` | (0, 0, 0) | center |
