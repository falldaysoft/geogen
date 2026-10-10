# trashcan

`asset: trashcan.yaml` · [source](../../../assets/trashcan.yaml) · `street.trashcan`

<img src="../img/trashcan.png" width="256" alt="trashcan"> <img src="../img/trashcan_views.png" width="512" alt="trashcan: front and side">

City street trash receptacle
Square bin with swing-top lid
Origin is at bottom_center

```yaml
place:
  trashcan: {asset: trashcan.yaml, params: {width: 0.45, height: 1}}
```

## Params

| Param | Default | Choices | Notes |
| --- | --- | --- | --- |
| `width` | `0.45` |  |  |
| `height` | `1` |  |  |
| `depth` | `0.45` |  |  |

## Size

0.45 x 1 x 0.45 m (x, y, z); geometry (-0.225, 0, -0.225) to (0.225, 1.09, 0.242) from the origin; 1384 triangles

## Parts

`base` (cube), `body_lower` (cube), `body_upper` (cube), `rim` (cube), `lid` (sphere), `opening_frame` (cube)
