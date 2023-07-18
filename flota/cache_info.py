"""Wrapper for cache info."""

from typing import NamedTuple


class CacheInfo(NamedTuple):
    """Representation wrapper for `functools.lru_cache` info."""

    hits: int
    misses: int
    maxsize: int | None
    currsize: int
