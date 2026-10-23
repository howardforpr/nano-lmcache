"""Recency policies rank entries; the index decides which leaves are safe."""

from typing import Protocol


class EvictionPolicy(Protocol):
    """Map an access tick to a heap priority; smaller values go first.

    A policy must be stateless: the same tick always has the same priority.
    Prefix safety, capacity, and stale heap records belong to the index.
    """

    def priority(self, last_access: int) -> int:
        """Rank an entry by its most recent successful put or get."""
        ...


class LRU:
    """Evict the least recently used eligible leaf (the default)."""

    def priority(self, last_access: int) -> int:
        return last_access


class MRU:
    """Evict the most recently used eligible leaf, retaining older prefixes."""

    def priority(self, last_access: int) -> int:
        return -last_access
