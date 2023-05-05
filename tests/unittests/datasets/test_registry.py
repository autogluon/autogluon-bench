from typing import OrderedDict as t_OrderedDict
from unittest.mock import MagicMock

import pytest

from autogluon.bench.datasets.registry import Registry


class MyClass:
    pass


class TestRegistry:
    def test_create_registry(self):
        registry = Registry("test")
        assert registry._name == "test"
        assert isinstance(registry._obj_map, t_OrderedDict)
        assert len(registry._obj_map) == 0

    def test_register(self):
        registry = Registry("test")
        registry.register(MyClass)
        assert len(registry._obj_map) == 1
        assert list(registry._obj_map.keys())[0] == "MyClass"
        assert registry.get("MyClass") == MyClass

    def test_register_with_nickname(self):
        registry = Registry("test")
        registry.register("alias", MyClass)
        assert len(registry._obj_map) == 1
        assert list(registry._obj_map.keys())[0] == "alias"
        assert registry.get("alias") == MyClass

    def test_get_non_existing_class(self):
        registry = Registry("test")
        with pytest.raises(KeyError):
            registry.get("NonExistingClass")

    def test_list_keys(self):
        registry = Registry("test")
        registry.register(MyClass)
        registry.register("alias", MagicMock())
        keys = registry.list_keys()
        assert "MyClass" in keys
        assert "alias" in keys

    def test_create_class(self):
        registry = Registry("test")
        registry.register(MyClass)
        obj = registry.create("MyClass")
        assert isinstance(obj, MyClass)
