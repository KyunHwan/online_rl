"""Independent registry implementation for policy components."""
from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A simple key -> object registry with optional base-class enforcement."""

    def __init__(self, name: str, expected_base: type | tuple[type, ...] | None = None) -> None:
        self._name = name
        self._expected_base = expected_base
        self._items: dict[str, T] = {}

    def register(self, key: str | None = None) -> Callable[[type[T]], type[T]]:
        """Decorator to register a class under a key."""

        def decorator(cls: type[T]) -> type[T]:
            reg_key = key or cls.__name__
            self.add(reg_key, cls)
            return cls

        return decorator

    def add(self, key: str, obj: T) -> None:
        if key in self._items:
            raise KeyError(f"{self._name} registry already has key '{key}'")
        self._enforce_expected_base(obj)
        self._items[key] = obj

    def get(self, key: str) -> T:
        if key not in self._items:
            raise KeyError(f"{self._name} registry has no key '{key}'. Available: {sorted(self._items)}")
        return self._items[key]

    def has(self, key: str) -> bool:
        return key in self._items

    def keys(self) -> list[str]:
        return sorted(self._items.keys())

    def _enforce_expected_base(self, obj: T) -> None:
        if self._expected_base is None:
            return
        expected = self._expected_base
        if isinstance(obj, type):
            if not issubclass(obj, expected):
                raise TypeError(
                    f"{self._name} registry expects subclasses of {expected}, got {obj}"
                )
        else:
            if not isinstance(obj, expected):
                raise TypeError(
                    f"{self._name} registry expects instances of {expected}, got {type(obj)}"
                )
