"""In-memory key-value cache, standing in for a real KV-tensor store."""

import sys
from collections.abc import Callable
from dataclasses import replace
from threading import Lock

from nano_lmcache.storage._lifetime import defer_release
from nano_lmcache.storage.eviction import PrefixLRUIndex
from nano_lmcache.storage.stats import StorageStats


class MemoryKVCache:
    """Thread-safe, optionally bounded cache storing opaque chunk payloads.

    Payloads are opaque `object` values here on purpose: this stage models
    the storage/lookup contract (put/get by key) without binding to any
    particular KV-tensor representation (e.g. PyTorch tensors).

    Capacity can be bounded by entries, payload bytes, or both. When full, the
    least recently used *leaf* is evicted, following the prefix-safe rules
    in `PrefixLRUIndex`: evicting an entry that still has descendants would
    strand them, since a prefix lookup could never reach them again.

    `size_fn` measures each payload on `put`, defaulting to the shallow
    `sys.getsizeof`. For tensors or containers of bytes, that measures the
    wrapper, not the backing data: pass a custom `size_fn` to enforce a
    meaningful payload budget. Keys and cache bookkeeping are excluded.
    Do not resize cached payloads in place;
    sizes are snapshots, refreshed only when the payload is put again.
    """

    def __init__(
        self,
        max_entries: int | None = None,
        *,
        max_bytes: int | None = None,
        size_fn: Callable[[object], int] = sys.getsizeof,
    ) -> None:
        self._index = PrefixLRUIndex(max_entries, max_bytes=max_bytes)
        self._size_fn = size_fn
        self._values: dict[str, object] = {}
        self._lock = Lock()

    @property
    def max_entries(self) -> int | None:
        """Entry limit, or None when the cache is unbounded."""
        return self._index.max_entries

    @property
    def max_bytes(self) -> int | None:
        """Payload byte limit, or None when unbounded."""
        return self._index.max_bytes

    @property
    def nbytes(self) -> int:
        """Sum of payload sizes measured at their last successful put."""
        with self._lock:
            return self._index.nbytes

    @property
    def stats(self) -> StorageStats:
        """A snapshot of eviction and rejection totals for this cache."""
        with self._lock:
            return replace(self._index.stats)

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)

    def __contains__(self, key: object) -> bool:
        """Whether `key` is stored; unlike `get`, does not count as a use."""
        with self._lock:
            return key in self._values

    def parent_of(self, key: str) -> str | None:
        """The parent recorded for a stored `key`; KeyError if absent."""
        with self._lock:
            if key not in self._values:
                raise KeyError(key)
            return self._index.parent_of(key)

    def put(
        self,
        key: str,
        value: object,
        *,
        parent_key: str | None = None,
    ) -> None:
        self.try_put(key, value, parent_key=parent_key)

    def try_put(
        self, key: str, value: object, *, parent_key: str | None = None
    ) -> bool:
        """Store if admitted; return False without replacing a rejected value."""
        # Measure and validate before changing recency, values, or capacity.
        # This also measures writes later rejected for a missing parent.
        size_bytes = self._size_fn(value)
        with defer_release() as retired, self._lock:
            decision = self._index.admit(key, parent_key, size_bytes=size_bytes)
            retired.extend(self._values.pop(victim) for victim in decision.evicted)

            if decision.admitted:
                retired.append(self._values.get(key))
                self._values[key] = value
        return decision.admitted

    def invalidate(self, key: str) -> None:
        """Remove a stale entry and its descendants, preserving prefix safety."""
        with defer_release() as retired, self._lock:
            retired.extend(
                self._values.pop(victim) for victim in self._index.invalidate(key)
            )

    def get(self, key: str) -> object | None:
        with self._lock:
            if key not in self._values:
                return None

            self._index.touch(key)
            return self._values[key]
