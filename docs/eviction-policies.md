# Recency policies (M11)

`PrefixLRUIndex` decides which entries are safe to evict; an
`EvictionPolicy` only ranks them. LRU (the default) ranks by last access
tick, and MRU ranks by its negative. Both preserve ancestors, protect the
incoming chain, and reject impossible writes before evicting anything.

This follows LMCache's separation of cache policy from storage, including
its LRU/MRU hit-ordering semantics. The implementation is independent:
nano-lmcache retains its existing leaf heap and adds a single stateless
`priority(last_access)` method, without a policy registry or configuration
framework. LFU and depth-aware policies would need additional metadata;
they are not supported by this recency-only interface.

All three bounded backends accept `policy=LRU()` or `policy=MRU()`.
Disk entries already persist access times, so recovery works for both
without a file-format change. `examples/demo_prefix_cache.py` demonstrates
the difference after accessing one of two cached tails.

## Shared system prompts and diverse tails

Measured on 2026-10-23 with Python 3.11.17, Linux x86_64, Intel Xeon
Platinum 8352Y at 2.20 GHz. No model, GPU, or PyTorch is needed.

```bash
uv run python benchmarks/bench_prefix_cache.py --policy-only --capacity 32 64 128
```

Each trace starts cold and contains 2000 requests of 12 chunks each,
with 16 tokens per chunk. Two distinct 4-chunk system prompts each serve
many 8-chunk tails. A tail's ID determines its system prompt. Every request
looks up its prefix, then writes only the missed suffix with `start_chunk`.
The same trace is replayed for each policy and capacity.

- **hot:** four popular tails receive 80% of requests; the rest have
  never-before-seen tails. A local `random.Random(0)` fixes request order.
- **scan:** requests cycle through 32 tails. The complete working set is
  264 entries, larger than any bounded capacity below.

Hit rate is `hit_chunks / total_chunks`, including cold misses and all
chunks after the first miss. Each row has 24000 requested chunks.

| Workload | Capacity | LRU hits | LRU hit rate | MRU hits | MRU hit rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| hot | unbounded | 20784 | 86.60% | 20784 | 86.60% |
| hot | 32 | 15272 | 63.63% | 10576 | 44.07% |
| hot | 64 | 20112 | 83.80% | 10664 | 44.43% |
| hot | 128 | 20784 | 86.60% | 10832 | 45.13% |
| scan | unbounded | 23736 | 98.90% | 23736 | 98.90% |
| scan | 32 | 7992 | 33.30% | 9016 | 37.57% |
| scan | 64 | 7992 | 33.30% | 11064 | 46.10% |
| scan | 128 | 7992 | 33.30% | 15160 | 63.17% |

LRU wins on the hot trace: recent accesses predict further reuse, so
discarding the newest leaf repeatedly loses useful tail chunks. MRU wins
on the scan trace: recently visited tails will not be revisited until
after the rest of the cycle, and retaining older tails avoids some of
LRU's repeated misses. Neither policy evicts shared internal prefixes.
These synthetic traces illustrate locality tradeoffs, not production
traffic or a universal winner; LRU remains the default.

## Default-LRU timing check

The same machine ran the following command before (upstream `50b4b4a`)
and after the ranking refactor, using the best of three repeats. Timed
store/lookup and churn workloads are unchanged. The policy workload runs
only with `--policy-only`, so neither timing run executes it as a warmup.

```bash
uv run python benchmarks/bench_prefix_cache.py --chunks 100 1000 \
  --capacity 64 --repeats 3 --churn-prompts 200 --churn-chunks 8
```

| Chunks | Capacity | Store before (ms) | Store after (ms) | Lookup before (ms) | Lookup after (ms) | Matched (both) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | unbounded | 1.100 | 1.096 | 0.799 | 0.811 | 100 |
| 100 | 64 | 1.206 | 1.093 | 0.581 | 0.549 | 64 |
| 1000 | unbounded | 15.388 | 10.802 | 8.041 | 8.520 | 1000 |
| 1000 | 64 | 10.589 | 9.246 | 0.798 | 0.809 | 64 |

| Churn workload | Capacity | Before (ms) | After (ms) | Entries kept (both) |
| --- | ---: | ---: | ---: | ---: |
| 200 prompts, 8 chunks each | unbounded | 18.484 | 18.577 | 1600 |
| 200 prompts, 8 chunks each | 64 | 26.535 | 27.781 | 64 |
| 64 prompts, 1024 chunks each | 8192 | 4480.947 | 4622.070 | 8192 |

The heap now stores `(priority, access_tick, key)`. The separate tick
invalidates stale records even when a custom policy produces equal
priorities. This adds a small tuple field and a method call; it is not a
performance optimization. These shared-machine timings fluctuate and
do not establish a speedup. Candidate maintenance remains O(log n)
amortized with O(n) retained heap records, as before.

## Correctness

The existing bounded-memory, index, and leaf-eviction tests are unchanged.
New randomized tests compare LRU and MRU to an independent safe scan
under entry limits, byte limits, and both together (three seeds, 600
operations per scenario). They check every admission and victim, capacity,
recency, and parent relation. Targeted tests cover rejected writes, stale
records, equal priorities, subtree removal, reinsertion, heap compaction,
backend selection, and disk restart. CPU-offload selection runs with
`device="cpu", pin_memory=False`.
