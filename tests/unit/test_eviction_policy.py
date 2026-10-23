"""Policy ranking may change, but admission must always preserve prefixes."""

import random

import pytest

from nano_lmcache.storage.disk import DiskKVCache
from nano_lmcache.storage.eviction import Admission, PrefixLRUIndex
from nano_lmcache.storage.memory import MemoryKVCache
from nano_lmcache.storage.policy import LRU, MRU


@pytest.mark.parametrize("policy,victim", [(LRU(), "b"), (MRU(), "a")])
@pytest.mark.parametrize("backend", ["memory", "disk"])
def test_backends_apply_policy_after_a_hit(tmp_path, backend, policy, victim):
    storage = (
        MemoryKVCache(max_entries=2, policy=policy)
        if backend == "memory"
        else DiskKVCache(tmp_path, max_entries=2, policy=policy)
    )
    storage.put("a", 1)
    storage.put("b", 2)
    assert storage.get("a") == 1
    storage.put("c", 3)

    assert victim not in storage
    assert len(storage) == 2


@pytest.mark.parametrize("policy,victim", [(LRU(), "b"), (MRU(), "a")])
def test_disk_recovery_preserves_policy_order(tmp_path, policy, victim):
    storage = DiskKVCache(tmp_path, max_entries=2, policy=policy)
    storage.put("a", 1)
    storage.put("b", 2)
    storage.get("a")

    reopened = DiskKVCache(tmp_path, max_entries=2, policy=policy)
    reopened.put("c", 3)

    assert victim not in reopened
    assert len(reopened) == 2


@pytest.mark.torch
@pytest.mark.parametrize("policy,victim", [(LRU(), "b"), (MRU(), "a")])
def test_cpu_offload_applies_policy(policy, victim):
    torch = pytest.importorskip("torch")
    from nano_lmcache.tensor.offload import CPUOffloadKVCache

    storage = CPUOffloadKVCache(
        device="cpu", pin_memory=False, max_entries=2, policy=policy
    )
    for key in "ab":
        storage.put(key, torch.ones(2))
    storage.get("a")
    storage.put("c", torch.ones(2))

    assert victim not in storage
    assert len(storage) == 2


@pytest.mark.parametrize("policy", [LRU(), MRU()])
def test_policy_cannot_evict_the_incoming_chain(policy):
    index = PrefixLRUIndex(max_bytes=4, policy=policy)
    index.admit("a", None, size_bytes=2)
    index.admit("b", "a", size_bytes=2)

    assert index.admit("c", "b", size_bytes=1) == Admission(False, ())
    assert index.admit("a", None, size_bytes=5) == Admission(False, ())
    assert set(index.keys_by_recency()) == {"a", "b"}
    assert index.nbytes == 4


def test_mru_exposed_parent_retains_its_actual_access_time():
    index = PrefixLRUIndex(max_entries=3, policy=MRU())
    index.admit("a", None)
    index.admit("x", None)
    index.admit("b", "a")
    assert index.admit("c", "x").evicted == ("b",)
    # a is newly exposed but still older than c, and c protects x.
    assert index.admit("d", "c").evicted == ("a",)
    assert index.admit("e", None).evicted == ("d",)


@pytest.mark.parametrize("policy", [LRU(), MRU()])
@pytest.mark.parametrize("limits", [(1, None), (8, None), (None, 12), (5, 12)])
@pytest.mark.parametrize("seed", [0, 17, 42])
def test_random_operations_match_safe_scan(policy, limits, seed):
    max_entries, max_bytes = limits
    index = PrefixLRUIndex(
        max_entries=max_entries, max_bytes=max_bytes, policy=policy
    )
    rng = random.Random(seed)
    # Independent scan model: dictionary insertion order is true recency.
    parents = {}
    sizes = {}
    keys = [str(i) for i in range(16)]

    def fits(entries, nbytes):
        return (max_entries is None or entries <= max_entries) and (
            max_bytes is None or nbytes <= max_bytes
        )

    for _ in range(600):
        key = rng.choice(keys)
        if key in parents and rng.randrange(3) == 0:
            index.touch(key)
            parents[key] = parents.pop(key)
        else:
            parent = parents.get(key) if key in parents else rng.choice([None, *keys])
            size = rng.randrange(1, 8)
            chain = [key]
            ancestor = parent
            while ancestor in parents:
                chain.append(ancestor)
                ancestor = parents[ancestor]
            possible = (parent is None or parent in parents) and fits(
                len(chain), size + sum(sizes[k] for k in chain[1:])
            )
            victims = []
            if possible:
                added_entries = int(key not in parents)
                added_bytes = size - sizes.get(key, 0)
                while not fits(
                    len(parents) + added_entries, sum(sizes.values()) + added_bytes
                ):
                    eligible = [
                        k for k in parents
                        if k not in chain and k not in parents.values()
                    ]
                    victim = eligible[0] if isinstance(policy, LRU) else eligible[-1]
                    victims.append(victim)
                    del parents[victim], sizes[victim]
                parents.pop(key, None)
                parents[key] = parent
                sizes[key] = size
            assert index.admit(key, parent, size_bytes=size) == Admission(
                possible, tuple(victims)
            )

        assert list(index.keys_by_recency()) == list(parents)
        assert index.nbytes == sum(sizes.values())
        assert fits(len(index), index.nbytes)
        for stored, parent in parents.items():
            assert index.parent_of(stored) == parent
            assert parent is None or parent in index


@pytest.mark.parametrize("policy", [LRU(), MRU()])
def test_removal_reinsertion_and_repeated_hits_keep_prefixes_and_heap_bounded(policy):
    index = PrefixLRUIndex(max_entries=4, policy=policy)
    index.admit("a", None)
    index.admit("b", "a")
    index.admit("c", "b")
    index.admit("x", None)
    assert index.discard_subtree("b") == ("c", "b")
    index.admit("b", "x")
    index.admit("c", "b")
    for _ in range(1000):
        index.touch("c")
    assert len(index._leaf_heap) <= 64
    assert index.admit("d", None).evicted == (
        "a" if isinstance(policy, LRU) else "c",
    )


def test_custom_equal_priorities_still_discard_stale_access_records():
    class EqualPriority:
        def priority(self, last_access):
            return 0

    index = PrefixLRUIndex(max_entries=2, policy=EqualPriority())
    index.admit("a", None)
    index.admit("b", None)
    index.touch("a")
    assert index.admit("c", None).evicted == ("b",)
