"""Tests for the YAML expression evaluator and param interpolation."""

import numpy as np
import pytest

from geogen.layout import LayoutLoader
from geogen.layout.expressions import (
    ExpressionError,
    evaluate,
    interpolate_string,
    resolve_params,
    resolve_value,
)


class TestEvaluate:
    def test_numeric_literal(self):
        assert evaluate("3", {}) == 3
        assert evaluate("0.5", {}) == 0.5
        assert evaluate("-2.5", {}) == -2.5

    def test_param_reference(self):
        assert evaluate("width", {"width": 6.0}) == 6.0

    def test_arithmetic(self):
        params = {"width": 6.0, "thickness": 0.2}
        assert evaluate("width - thickness", params) == pytest.approx(5.8)
        assert evaluate("width * 0.5", params) == 3.0
        assert evaluate("width / 2", params) == 3.0
        assert evaluate("(width - thickness) * 0.5", params) == pytest.approx(2.9)

    def test_unary(self):
        assert evaluate("-3", {}) == -3
        assert evaluate("-width", {"width": 5.0}) == -5.0

    def test_unknown_param_raises(self):
        with pytest.raises(ExpressionError, match="Unknown parameter"):
            evaluate("foo + 1", {"width": 5})

    def test_disallows_function_calls(self):
        with pytest.raises(ExpressionError):
            evaluate("len(width)", {"width": 5})

    def test_disallows_attribute_access(self):
        with pytest.raises(ExpressionError):
            evaluate("__import__('os').system('ls')", {})

    def test_unit_literal_cm(self):
        assert evaluate("50cm", {}) == pytest.approx(0.5)

    def test_unit_literal_m(self):
        assert evaluate("2.5m", {}) == pytest.approx(2.5)

    def test_unit_literal_percent(self):
        assert evaluate("20%", {}) == pytest.approx(0.2)


class TestInterpolateString:
    def test_whole_interp_returns_float(self):
        result = interpolate_string("{width}", {"width": 6.0})
        assert result == 6.0
        assert isinstance(result, float)

    def test_bare_unit_literal_returns_float(self):
        assert interpolate_string("50cm", {}) == pytest.approx(0.5)

    def test_embedded_interp_returns_string(self):
        result = interpolate_string("prefix_{width}_suffix", {"width": 6})
        assert result == "prefix_6.0_suffix"

    def test_plain_string_unchanged(self):
        assert interpolate_string("hello", {}) == "hello"


class TestResolveValue:
    def test_list_with_interpolations(self):
        params = {"w": 8.0, "h": 4.0, "d": 6.0}
        result = resolve_value(["{w}", "{h}", "{d}"], params)
        assert result == [8.0, 4.0, 6.0]

    def test_nested_dict(self):
        params = {"width": 6.0}
        result = resolve_value(
            {"size": ["{width}", 1, "{width / 2}"]}, params
        )
        assert result == {"size": [6.0, 1, 3.0]}

    def test_numbers_pass_through(self):
        assert resolve_value(42, {}) == 42
        assert resolve_value(3.14, {}) == 3.14

    def test_none_passes_through(self):
        assert resolve_value(None, {}) is None


class TestResolveParams:
    def test_defaults_applied(self):
        declared = {"width": {"default": 6}, "height": {"default": 3}}
        assert resolve_params(declared, None) == {"width": 6.0, "height": 3.0}

    def test_overrides_applied(self):
        declared = {"width": {"default": 6}}
        assert resolve_params(declared, {"width": 10}) == {"width": 10.0}

    def test_unknown_override_raises(self):
        declared = {"width": {"default": 6}}
        with pytest.raises(ExpressionError, match="Unknown param override 'depth'"):
            resolve_params(declared, {"depth": 5})

    def test_missing_default_raises(self):
        with pytest.raises(ExpressionError, match="requires a 'default'"):
            resolve_params({"width": {}}, None)

    def test_unit_string_default(self):
        declared = {"wall": {"default": "15cm"}}
        assert resolve_params(declared, None)["wall"] == pytest.approx(0.15)

    def test_empty_declared(self):
        assert resolve_params(None, None) == {}
        assert resolve_params({}, None) == {}


class TestLoaderParams:
    def test_parametric_asset_default(self):
        loader = LayoutLoader()
        yaml_str = """
name: box
params:
  width: { default: 6 }
  height: { default: 3 }
  depth: { default: 4 }
size: ["{width}", "{height}", "{depth}"]
parts:
  body:
    primitive: cube
    size: [1, 1, 1]
    anchor: bottom_center
"""
        node = loader.load_string(yaml_str)
        assert node.size is not None
        np.testing.assert_allclose(node.size, [6, 3, 4])

    def test_parametric_asset_override(self):
        loader = LayoutLoader()
        yaml_str = """
name: box
params:
  width: { default: 6 }
  height: { default: 3 }
  depth: { default: 4 }
size: ["{width}", "{height}", "{depth}"]
parts:
  body:
    primitive: cube
    size: [1, 1, 1]
    anchor: bottom_center
"""
        node = loader.load_string(yaml_str, params={"width": 10, "depth": 8})
        np.testing.assert_allclose(node.size, [10, 3, 8])

    def test_expression_in_part_offset(self):
        loader = LayoutLoader()
        yaml_str = """
name: box
params:
  wall: { default: 0.2 }
size: [1, 1, 1]
parts:
  body:
    primitive: cube
    size: ["{1 - wall}", 1, 1]
    anchor: bottom_center
"""
        node = loader.load_string(yaml_str)
        # The cube part has actual_size = frac_size * container_size.
        # Container is [1,1,1] and frac_size[0] is (1 - 0.2) = 0.8.
        body = node.children[0]
        assert body.size is not None
        assert body.size[0] == pytest.approx(0.8)

    def test_unit_suffix_in_yaml(self):
        loader = LayoutLoader()
        yaml_str = """
name: box
size: [1, 1, 1]
parts:
  body:
    primitive: cube
    size: ["50cm", "20%", 1]
    anchor: bottom_center
"""
        node = loader.load_string(yaml_str)
        body = node.children[0]
        assert body.size is not None
        # 50cm → 0.5 frac * container 1 = 0.5
        assert body.size[0] == pytest.approx(0.5)
        # 20% → 0.2 frac * container 1 = 0.2
        assert body.size[1] == pytest.approx(0.2)
