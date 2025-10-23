"""Light-weight cache helpers for experiment artifacts."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CacheManager:
    """Simple directory-based cache for binary artifacts."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def has(self, relative_path: str) -> bool:
        return self.path(relative_path).exists()

    def load_pickle(self, relative_path: str) -> Any:
        with open(self.path(relative_path), "rb") as fh:
            return pickle.load(fh)

    def save_pickle(self, relative_path: str, obj: Any) -> None:
        target = self.path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as fh:
            pickle.dump(obj, fh)
