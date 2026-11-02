from __future__ import annotations

from typing import Any

COMPOSITIONS = {"OneOf", "SomeOf", "Sequential"}


def _tuplify(v: Any) -> Any:
    return tuple(v) if isinstance(v, list) and all(isinstance(x, int | float) for x in v) else v


def build_transform(spec: dict[str, Any]):
    import albumentations as A

    if len(spec) != 1:
        raise ValueError(f"each augmentation must be a single-key mapping, got {spec}")
    ((name, kwargs),) = spec.items()
    kwargs = dict(kwargs or {})
    cls = getattr(A, name, None)
    if cls is None:
        raise ValueError(f"unknown Albumentations transform {name!r}")
    if name in COMPOSITIONS:
        children = [build_transform(c) for c in kwargs.pop("transforms")]
        return cls(children, **{k: _tuplify(v) for k, v in kwargs.items()})
    return cls(**{k: _tuplify(v) for k, v in kwargs.items()})


def build_augmentations(specs: list[dict[str, Any]] | None) -> list | None:
    if specs is None:
        return None
    return [build_transform(s) for s in specs]
