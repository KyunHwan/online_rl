"""Plugin loader for registry extensions."""
from __future__ import annotations

import importlib
from typing import Iterable

_LOADED_MODULES: set[str] = set()


def load_plugins(modules: Iterable[str]) -> None:
    """Import extension modules once to register custom components."""
    for module in modules:
        if module in _LOADED_MODULES:
            continue
        importlib.import_module(module)
        _LOADED_MODULES.add(module)
