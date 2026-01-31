"""Policy interface for env_actor."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class Policy(Protocol):
    """Interface for inference-time policies built from component modules."""

    def __init__(self, components: dict[str, nn.Module], **kwargs: Any) -> None: ...
