import random
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import Barrier, Thread

import pytest

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.storage.disk import DiskKVCache
from nano_lmcache.storage.eviction import PrefixLRUIndex
from nano_lmcache.storage.memory import MemoryKVCache
from nano_lmcache.storage.tiered import TieredStorage


@pytest.mark.parametrize("backend", ["memory", "disk", "tiered"])
def test_concurrent_prefix_requests_preserve_storage(backend, tmp_path, monkeypatch):
    memory = MemoryKVCache(max_entries=13, max_bytes=72, size_fn=len)
    disk = DiskKVCache(tmp_path, max_entries=13)
    storage = {
        "memory": memory,
        "disk": disk,
        "tiered": TieredStorage(fast=memory, slow=disk),
    }[backend]
    admit = PrefixLRUIndex.admit

    def yielding_admit(self, *args, **kwargs):
        decision = admit(self, *args, **kwargs)
        # Yield after the index changes, before the backend moves payloads.
        # Without a backend lock, another operation sees incomplete state.
        time.sleep(0)
        return decision

    monkeypatch.setattr(PrefixLRUIndex, "admit", yielding_admit)
    workers, operations, chunks = 8, 80, 4
    start = Barrier(workers)

    def request(prompt):
        return [prompt, 0, 1, 2, 3, 4, 5, 6]

    def run(worker):
        rng = random.Random(worker)
        cache = ChunkedPrefixCache(chunk_size=2, namespace="threads", storage=storage)
        stored = []
        lookups = probes = hits = 0
        start.wait(timeout=10)
        for step in range(operations):
            if rng.random() < 0.6:
                # Each prompt is written only once, making write accounting exact.
                prompt = worker * operations + step
                cache.store(request(prompt), [f"{prompt}:{i}" for i in range(chunks)])
                stored.append(prompt)
            else:
                prompt = (
                    rng.choice(stored)
                    if stored else rng.randrange(workers * operations)
                )
                result = cache.lookup(request(prompt))
                assert result.values == tuple(
                    f"{prompt}:{i}" for i in range(result.matched_chunks)
                )
                assert result.matched_tokens == result.matched_chunks * 2
                lookups += 1
                hits += result.matched_chunks
                probes += result.matched_chunks + (result.matched_chunks < chunks)

        assert cache.metrics.total_chunks == lookups * chunks
        assert cache.metrics.hit_chunks == hits
        assert cache.metrics.hit_tokens == hits * 2
        return len(stored) * chunks, probes

    with ThreadPoolExecutor(max_workers=workers) as pool:
        totals = list(pool.map(run, range(workers)))

    tiers = (memory, disk) if backend == "tiered" else (storage,)
    for tier in tiers:
        assert len(tier) <= tier.max_entries
        keys = set(tier._index.keys_by_recency())
        for key in keys:
            parent = tier.parent_of(key)
            assert parent is None or parent in keys
            assert tier.get(key) is not None
        if isinstance(tier, MemoryKVCache):
            assert keys == tier._values.keys()
            assert tier.nbytes == sum(map(len, tier._values.values()))
            assert tier.nbytes <= tier.max_bytes
        else:
            assert len(list(tmp_path.glob("*.entry"))) == len(tier)
            assert not list(tmp_path.glob("*.tmp"))
            assert len(DiskKVCache(tmp_path, max_entries=13)) == len(tier)
        assert tier.stats.corrupted_entries == 0

    # Each successful unique insertion is either still present or was evicted.
    counted = disk if backend == "tiered" else storage
    assert sum(writes for writes, _ in totals) == (
        len(counted) + counted.stats.evictions + counted.stats.rejected
    )
    if isinstance(storage, TieredStorage):
        stats = storage.tier_stats
        assert stats.lookups == sum(probes for _, probes in totals)
        assert 0 <= stats.promotions <= stats.slow_hits


@pytest.mark.parametrize("backend", ["memory", "disk", "tiered"])
def test_statistics_are_snapshots(backend, tmp_path):
    storage = {
        "memory": MemoryKVCache(max_entries=1),
        "disk": DiskKVCache(tmp_path, max_entries=1),
        "tiered": TieredStorage(fast=MemoryKVCache(), slow=MemoryKVCache()),
    }[backend]
    if isinstance(storage, TieredStorage):
        before = storage.tier_stats
        storage.get("missing")
        assert before.misses == 0
        assert storage.tier_stats.misses == 1
    else:
        before = storage.stats
        storage.put("a", "value")
        storage.put("b", "value")
        assert before.evictions == 0
        assert storage.stats.evictions == 1


def test_concurrent_overwrites_keep_tiers_consistent(monkeypatch):
    fast, slow = MemoryKVCache(), MemoryKVCache()
    storage = TieredStorage(fast=fast, slow=slow)
    put = slow.put

    def yielding_put(*args, **kwargs):
        put(*args, **kwargs)
        # A second write must not pass the first between its two tier updates.
        time.sleep(0)

    monkeypatch.setattr(slow, "put", yielding_put)
    start = Barrier(8)

    def run(worker):
        start.wait(timeout=10)
        for step in range(40):
            storage.put("shared", (worker, step))
            assert storage.get("shared") is not None

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, range(8)))

    assert fast.get("shared") == slow.get("shared")
    assert storage.tier_stats.lookups == 320


@pytest.mark.parametrize("operation", ["evict", "overwrite", "invalidate"])
def test_payload_finalizer_runs_after_memory_update(operation):
    storage = MemoryKVCache(
        max_entries=2 if operation == "invalidate" else 1,
        size_fn=lambda value: 1,
    )
    observed = []

    class Payload:
        def __del__(self):
            observed.append((len(storage), storage.stats.evictions, storage.nbytes))

    storage.put("old", Payload())
    if operation == "invalidate":
        storage.put("child", Payload(), parent_key="old")

    def update():
        if operation == "invalidate":
            storage.invalidate("old")
        else:
            storage.put("new" if operation == "evict" else "old", "replacement")

    # A regression must fail promptly rather than leave pytest waiting forever.
    worker = Thread(target=update, daemon=True)
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive(), "payload finalizer blocked on the storage lock"
    expected = {
        "evict": (1, 1, 1),
        "overwrite": (1, 0, 1),
        "invalidate": (0, 2, 0),
    }
    assert observed == [expected[operation]] * (2 if operation == "invalidate" else 1)


@pytest.mark.parametrize(
    "operation", ["overwrite", "slow_overwrite", "evict", "invalidate", "promote"]
)
@pytest.mark.parametrize("callback", ["read", "write"])
def test_payload_finalizer_runs_after_both_tiers_update(operation, callback):
    fast = MemoryKVCache(max_entries=1, max_bytes=1, size_fn=len)
    slow = MemoryKVCache(max_entries=2 if operation == "promote" else 1)
    storage = TieredStorage(fast=fast, slow=slow)
    watched = "old" if "overwrite" in operation or operation == "invalidate" else "new"
    observed = []

    class Payload:
        def __len__(self):
            return 1

        def __del__(self):
            # The wrapper must be unlocked and both tiers must have finished.
            observed.append((fast.get(watched), slow.get(watched), storage.tier_stats))
            if callback == "write":
                storage.put("callback", "c")

    storage.put("old", Payload())
    if operation == "slow_overwrite":
        fast.invalidate("old")  # The slow-tier overwrite releases the last copy.
    elif operation == "promote":
        slow.put("new", "n")
        slow.put("other", "o")  # Only the fast tier still holds the payload.

    def update():
        if operation == "promote":
            storage.get("new")
        else:
            storage.put(watched, "nn" if operation == "invalidate" else "n")

    worker = Thread(target=update, daemon=True)
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive(), "payload finalizer blocked on the tiered lock"
    assert len(observed) == 1
    assert observed[0][:2] == (
        None if operation == "invalidate" else "n",
        "nn" if operation == "invalidate" else "n",
    )
    assert observed[0][2].promotions == (operation == "promote")
    if callback == "write":
        assert fast.get("callback") == slow.get("callback") == "c"


def test_nested_tiers_defer_finalizers_until_outer_wrapper_unlocks():
    inner = TieredStorage(fast=MemoryKVCache(), slow=MemoryKVCache())
    storage = TieredStorage(fast=inner, slow=MemoryKVCache())
    observed = []

    class Payload:
        def __del__(self):
            observed.append((storage.tier_stats, inner.tier_stats))
            storage.put("callback", "done")

    storage.put("key", Payload())
    worker = Thread(target=lambda: storage.put("key", "new"), daemon=True)
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive(), "payload finalizer blocked on an outer tier lock"
    assert len(observed) == 1
    assert storage.get("callback") == inner.slow.get("callback") == "done"


def test_tiered_write_error_releases_retired_payloads_after_unlock(monkeypatch):
    fast, slow = MemoryKVCache(), MemoryKVCache()
    storage = TieredStorage(fast=fast, slow=slow)
    observed = []
    errors = []

    class Payload:
        def __del__(self):
            observed.append(storage.tier_stats)

    storage.put("key", Payload())
    fast.invalidate("key")

    def fail(*args, **kwargs):
        raise ValueError("fast tier failed")

    monkeypatch.setattr(fast, "try_put", fail)

    def update():
        try:
            storage.put("key", "new")
        except ValueError as error:
            errors.append(str(error))

    worker = Thread(target=update, daemon=True)
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive(), "error cleanup blocked on the tiered lock"
    assert errors == ["fast tier failed"]
    assert len(observed) == 1


def test_copied_context_does_not_keep_retired_payloads_alive():
    contexts = []

    def size_fn(value):
        if not contexts:
            contexts.append(copy_context())
        return 1

    tiered = TieredStorage(fast=MemoryKVCache(size_fn=size_fn), slow=MemoryKVCache())
    tiered.put("seed", "seed")

    class Payload:
        pass

    storage = MemoryKVCache()
    storage.put("key", Payload())
    reference = weakref.ref(storage.get("key"))
    contexts[0].run(storage.put, "key", "new")
    assert reference() is None
