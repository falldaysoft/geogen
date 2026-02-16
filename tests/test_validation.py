"""Tests for YAML validation and error handling."""

import warnings

import pytest

from geogen.layout.validation import (
    validate_asset_yaml,
    validate_scene_yaml,
    pre_validate_asset_references,
    pre_validate_scene_references,
    ValidationError,
)
from geogen.layout import LayoutLoader, SceneComposer
from geogen.materials.loader import MaterialLoader


class TestAssetValidation:
    def test_valid_asset(self):
        data = {
            "name": "test",
            "size": [1, 1, 1],
            "parts": {
                "base": {
                    "primitive": "cube",
                    "size": [1, 1, 1],
                    "anchor": "bottom_center",
                }
            },
        }
        warnings_list = validate_asset_yaml(data)
        assert warnings_list == []

    def test_unknown_top_level_key(self):
        data = {
            "name": "test",
            "size": [1, 1, 1],
            "colour": "red",  # misspelled/unknown
        }
        warnings_list = validate_asset_yaml(data)
        assert any("colour" in w for w in warnings_list)

    def test_missing_size(self):
        data = {"name": "test", "parts": {}}
        with pytest.raises(ValidationError, match="requires 'size'"):
            validate_asset_yaml(data)

    def test_unknown_primitive(self):
        data = {
            "name": "test",
            "size": [1, 1, 1],
            "parts": {
                "base": {"primitive": "cuube", "size": [1, 1, 1]}
            },
        }
        with pytest.raises(ValidationError, match="cuube.*Did you mean 'cube'"):
            validate_asset_yaml(data)

    def test_unknown_part_key(self):
        data = {
            "name": "test",
            "size": [1, 1, 1],
            "parts": {
                "base": {
                    "primitive": "cube",
                    "size": [1, 1, 1],
                    "matrial": "wood",  # misspelled
                }
            },
        }
        warnings_list = validate_asset_yaml(data)
        assert any("matrial" in w for w in warnings_list)
        assert any("material" in w for w in warnings_list)


class TestSceneValidation:
    def test_valid_scene(self):
        data = {
            "name": "test",
            "size": [1, 1, 1],
            "slots": {
                "center": {"anchor": "center", "facing": "north"}
            },
            "place": {
                "obj": {"asset": "test.yaml", "slot": "center"}
            },
        }
        warnings_list = validate_scene_yaml(data)
        assert warnings_list == []

    def test_unknown_facing(self):
        data = {
            "name": "test",
            "slots": {
                "s1": {"anchor": "center", "facing": "nroth"}
            },
        }
        warnings_list = validate_scene_yaml(data)
        assert any("nroth" in w for w in warnings_list)

    def test_missing_asset_or_scene(self):
        data = {
            "name": "test",
            "place": {
                "obj": {"slot": "center"}  # missing asset/scene
            },
        }
        with pytest.raises(ValidationError, match="requires 'asset' or 'scene'"):
            validate_scene_yaml(data)


class TestReferencePreValidation:
    def test_bad_attach_to_in_asset(self):
        data = {
            "name": "test",
            "size": [1, 1, 1],
            "parts": {
                "base": {"primitive": "cube", "size": [1, 1, 1]},
                "child": {
                    "primitive": "cube",
                    "size": [0.5, 0.5, 0.5],
                    "attach_to": "baze",  # misspelled
                },
            },
        }
        errors = pre_validate_asset_references(data)
        assert len(errors) == 1
        assert "baze" in errors[0]
        assert "base" in errors[0]  # suggestion

    def test_bad_slot_in_scene(self):
        data = {
            "name": "test",
            "slots": {"center_slot": {}},
            "place": {
                "obj": {"asset": "test.yaml", "slot": "centr_slot"}
            },
        }
        errors = pre_validate_scene_references(data)
        assert len(errors) == 1
        assert "centr_slot" in errors[0]

    def test_bad_attach_to_in_scene(self):
        data = {
            "name": "test",
            "place": {
                "table": {"asset": "table.yaml"},
                "chair": {"asset": "chair.yaml", "attach_to": "tabel"},
            },
        }
        errors = pre_validate_scene_references(data)
        assert len(errors) == 1
        assert "tabel" in errors[0]


class TestMaterialErrorMessages:
    def test_material_not_found_suggests(self):
        loader = MaterialLoader()
        with pytest.raises(FileNotFoundError, match="Did you mean"):
            loader.load("wod")  # close to "wood"

    def test_material_not_found_lists_available(self):
        loader = MaterialLoader()
        with pytest.raises(FileNotFoundError, match="Available materials"):
            loader.load("nonexistent_material_xyz")


class TestLoaderErrorMessages:
    def test_bad_attachment_point_includes_available(self):
        loader = LayoutLoader()
        yaml_str = """
name: test
size: [1, 1, 1]
parts:
  base:
    primitive: cube
    size: [1, 1, 1]
    anchor: bottom_center
  child:
    primitive: sphere
    size: [0.5, 0.5, 0.5]
    attach_to: base
    at: topp
"""
        with pytest.raises(ValueError, match="topp.*Available"):
            loader.load_string(yaml_str)

    def test_misspelled_material_emits_warning(self):
        loader = LayoutLoader()
        yaml_str = """
name: test
size: [1, 1, 1]
parts:
  base:
    primitive: cube
    size: [1, 1, 1]
    anchor: bottom_center
    material: wod
"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loader.load_string(yaml_str)
            material_warnings = [x for x in w if "wod" in str(x.message)]
            assert len(material_warnings) >= 1
