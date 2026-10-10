# road

`asset: road.yaml` · [source](../../../assets/road.yaml) · `street.road`

<img src="../img/road.png" width="256" alt="road"> <img src="../img/road_views.png" width="512" alt="road: front and side">

Road surface - just the asphalt

```yaml
place:
  road: {asset: road.yaml, params: {width: 6, thickness: 0.1}}
```

## Params

| Param | Default | Choices | Notes |
| --- | --- | --- | --- |
| `width` | `6` |  | across the road |
| `thickness` | `0.1` |  | asphalt slab height |
| `length` | `24` |  | along the road |

## Size

6 x 0.1 x 24 m (x, y, z); geometry (-3, 0, -12) to (3, 0.1, 12) from the origin; 12 triangles

## Parts

`surface` (cube)
