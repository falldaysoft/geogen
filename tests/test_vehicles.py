"""Vehicle assets and the vehicle: block (geogen/vehicles.py)."""

import json
from pathlib import Path

import numpy as np
import pytest

from geogen.core import meshops
from geogen.export import export_scene
from geogen.layout.composer import SceneComposer
from geogen.layout.loader import LayoutLoader

ASSETS = Path(__file__).parent.parent / "assets"
VEHICLES = sorted((ASSETS / "vehicles").glob("*.yaml"))


@pytest.fixture(scope="module")
def loader():
    return LayoutLoader()


@pytest.mark.parametrize("path", VEHICLES, ids=lambda p: p.stem)
def test_vehicle_meta(loader, path):
    node = loader.load(path)
    v = node.meta["vehicle"]
    assert node.meta["type"] == "vehicle"
    assert len(v["wheels"]) == 4 and v["wheelbase"] > 2 and v["track"] > 1.2
    for w in v["wheels"]:
        # Wheels sit on the ground: centre height == radius.
        assert w["center"][1] == pytest.approx(w["radius"], abs=0.01)
    length, width, height = v["clearance"]
    assert v["front"] > 0 > v["rear"] and length == pytest.approx(v["front"] - v["rear"], abs=0.02)
    assert 1.5 < width < 3.0 and 1.2 < height < 3.5
    # Every mesh is closed and valid.
    for n, mesh in node.iter_meshes():
        assert not meshops.validate(mesh).issues, n.name
    # Paint parts carry the tint.
    for part in v["paint"]:
        mesh = node.find(part).mesh
        assert mesh.colors is not None and np.ptp(mesh.colors[:, :3], axis=0).max() < 1e-6


def test_car_presets_and_taxi_sign(loader):
    path = ASSETS / "vehicles" / "car.yaml"
    sedan = loader.load(path)
    hatch = loader.load(path, params={"preset": "hatchback"})
    taxi = loader.load(path, params={"preset": "taxi"})
    assert hatch.meta["vehicle"]["clearance"][0] < sedan.meta["vehicle"]["clearance"][0]
    assert sedan.find("taxi_sign") is None and taxi.find("taxi_sign") is not None
    yellow = taxi.find("body").mesh.colors[0]
    assert yellow[0] > 0.8 and yellow[2] < 0.2
    red = loader.load(path, params={"paint": "car_red"}).find("body").mesh.colors[0]
    assert red[0] > 0.5 and red[1] < 0.2


def test_export_renames_vehicle_parts(tmp_path):
    root = SceneComposer(assets_dir=ASSETS).compose_string(
        "name: two_cars\n"
        "place:\n"
        "  a: { asset: vehicles/car.yaml }\n"
        "  b: { asset: vehicles/car.yaml, params: { paint: car_blue } }\n")
    path = export_scene(root, tmp_path / "cars.glb")
    import struct
    data = path.read_bytes()
    gltf = json.loads(data[20:20 + struct.unpack("<I", data[12:16])[0]])
    names = {n["name"] for n in gltf["nodes"]}
    vehicles = [n["extras"]["geogen"]["vehicle"] for n in gltf["nodes"]
                if n.get("extras", {}).get("geogen", {}).get("vehicle")]
    assert len(vehicles) == 2
    wheels = [w["part"] for v in vehicles for w in v["wheels"]]
    assert len(set(wheels)) == 8 and set(wheels) <= names
