"""Docling integration. Names are resolved lazily: the adapter imports the candidates package, which imports the
Docling table provider, so an eager import here would be circular whenever ``app.services.docling`` is imported first."""
from __future__ import annotations

__all__ = ["docling_table_to_internal", "normalize_box"]


def __getattr__(name: str):
    if name in __all__:
        from . import adapter

        return getattr(adapter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
