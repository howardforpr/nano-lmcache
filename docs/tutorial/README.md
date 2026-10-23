# Start here: your first cache experiment

By the end of this chapter, you should be able to predict which chunks a
request reuses and explain the demo's hit/miss counters. You need Python
3.11 or later, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and a terminal. This experiment runs on a CPU with the core dependencies;
it does not need PyTorch, a GPU, a model download, or a running service.

## Set up and run

From a new checkout:

```bash
git clone https://github.com/Dami-wq-k03/nano-lmcache.git
cd nano-lmcache
uv sync --locked
uv run --locked python examples/demo_prefix_cache.py
```

If you already have a checkout, run the last two commands from its root.
`uv` creates a project-local `.venv` and installs the development tools.
The initial setup needs network access to download those tools; subsequent
demo runs use only local code and memory.

The demo stores strings such as `kv:101-205` in place of real KV tensors.
The input integers are illustrative token IDs, already past tokenization.
The printed word `recomputed` means the demo built a string; no language
model runs here, so these counters do not measure inference time.

## Read Part 1: cold, partially warm, and fully warm

Focus first on `Part 1: unbounded cache`. All four requests share one
in-memory cache, with two tokens per chunk:

```text
A:       [101,205 | 307,409 | 511,613]
B:       [101,205 | 307,409 | 777,888]
A again: [101,205 | 307,409 | 511,613]
C:       [101,205 | 307,409 | 511]
```

The vertical bars above mark chunks; they are explanatory notation, not
part of the Python lists. A request first looks up its prefix, then stores
payloads for its complete chunks so later requests can reuse them.

| Request | What is already cached? | Reused chunks | Reused tokens |
| --- | --- | ---: | ---: |
| A | Nothing: this is a cold cache | 0/3 | 0/6 |
| B | A's first two chunks; B's third chunk differs | 2/3 | 4/6 |
| A again | All three chunks of A remain cached | 3/3 | 6/6 |
| C | Its two complete chunks; the one-token tail is excluded | 2/2 | 4/5 |

For B, the relevant output is:

```text
--- Request B (shares 4-token prefix) ---
Tokens: [101, 205, 307, 409, 777, 888]
  chunk 0 (101, 205) HIT   reused 97005f3dd655...
  chunk 1 (307, 409) HIT   reused 8f3449bf7406...
  chunk 2 (777, 888) MISS  recomputed
  reused 2/3 chunks, 4/6 tokens
```

Cold and warm describe the cache's contents at the time of a request. B
is partially warm; the repeated A is fully warm. Running the script
again starts a new process and an empty memory cache, so the first A is
cold again. Persistence comes later in the disk-cache chapter.

Lookup walks from the first chunk and stops at the first miss. The key
for each chunk includes its preceding key, so the same tokens following
a different prefix cannot accidentally match. C's final `(511,)` is an
incomplete chunk: this demo neither stores it nor includes it in reuse
metrics. A real inference engine would still need to compute that token.

## Explain the counters

At the end of Part 1, the demo prints:

```text
=== Cumulative metrics ===
Hit chunks:  7
Miss chunks: 4
Hit tokens:  14
Hit rate:    63.6%
```

There are `3 + 3 + 3 + 2 = 11` eligible chunks across the four requests.
The hits are `0 + 2 + 3 + 2 = 7`, so the hit rate is `7 / 11`, rounded to
63.6%. Each hit covers two tokens, giving 14 hit tokens. These are totals
across requests, not the number of distinct items currently stored.

Every complete chunk after the first miss also counts as a miss, because
the caller still has to compute it. The backend is not probed for those
later chunks. An incomplete tail counts as neither a hit nor a miss.
The counters update on `lookup`, not on `store`.

Parts 2 and 3 create fresh caches with entry and byte limits, respectively;
their counters start from zero. A formerly warm request can lose its tail
after eviction. Follow the [capacity explanation](../architecture.md#capacity-and-eviction)
when you are ready to predict those sections too.

## Exercise: change the first token

Suppose A is stored and B changes its first token from `101` to `999`,
while the other tokens stay the same. Predict the matched chunks,
matched tokens, and chunk misses before running this from the repository
root in a shell that supports here-documents:

```bash
uv run --locked python - <<'PY'
from nano_lmcache.core.prefix_cache import ChunkedPrefixCache

cache = ChunkedPrefixCache(chunk_size=2, namespace="demo-model")
cache.store([101, 205, 307, 409, 511, 613], ["a0", "a1", "a2"])
result = cache.lookup([999, 205, 307, 409, 777, 888])
print("matched chunks:", result.matched_chunks)
print("matched tokens:", result.matched_tokens)
print("miss chunks:", cache.metrics.miss_chunks)
PY
```

Hint: matching must start at the beginning; finding an equal middle chunk
is not enough. The answer is `0`, `0`, and `3`. The first key differs, so
lookup stops immediately and all three complete chunks need computation.

For the implementation, read [serve_request](../../examples/demo_prefix_cache.py),
then [lookup and store](../../src/nano_lmcache/core/prefix_cache.py), and
[CacheMetrics](../../src/nano_lmcache/core/metrics.py). The
[prefix tests](../../tests/unit/test_prefix_cache.py) cover cold lookup,
partial hits, and incomplete tails; the
[metrics tests](../../tests/unit/test_metrics.py) cover counter arithmetic.
Run those tests with:

```bash
uv run --locked pytest tests/unit/test_prefix_cache.py tests/unit/test_metrics.py
```

## Where to go next

The follow-on chapters are proposed in the
[teaching roadmap](https://github.com/Dami-wq-k03/nano-lmcache/issues/63).
Until they land, the links below point to existing material:

| Topic | Current reading or experiment | Requirements |
| --- | --- | --- |
| Chunks and prefix-aware keys (EDU2) | [Chunking demo](../../examples/demo_chunking.py), [key diagram](../architecture.md#how-a-chunk-key-is-built) | CPU, core dependencies |
| Follow lookup and store (EDU3) | [Core code](../../src/nano_lmcache/core/prefix_cache.py), [metric explanation](../architecture.md#why-metrics-count-the-way-they-do) | CPU, core dependencies |
| Capacity and eviction (EDU4) | Parts 2 and 3 of this demo; [eviction rules](../architecture.md#capacity-and-eviction) | CPU, core dependencies |
| Persistence and tiers (EDU5) | [Disk demo](../../examples/demo_disk_cache.py), [tiered demo](../../examples/demo_tiered_storage.py), [inspector](../../src/nano_lmcache/inspect.py) | CPU, core dependencies |
| Real KV tensors (EDU6) | [Tensor explanation](../architecture.md#real-kv-tensors), [tiny-model tests](../../tests/tensor/test_hf_reuse.py) | CPU with optional tensor/HF dependencies |
| Measure reuse (EDU7) | [Core benchmark](../../benchmarks/bench_prefix_cache.py); [engine measurements](../nanovllm-benchmark.md) | Core benchmark: CPU; engine experiment: documented GPU setup |
| Inference engines (EDU8) | [nano-vLLM integration](../nanovllm-integration.md) | Reading: CPU; running: Linux, GPU, optional runtime and model |

Keep the [glossary and FAQ](../glossary.md) nearby for definitions of
token, chunk, prefix, prefill, and KV cache. nano-lmcache introduces these
concepts so you can later read [LMCache](https://github.com/LMCache/LMCache)'s
production implementation; the [project roadmap](../../ROADMAP.md)
explains the scope of that comparison.

## Troubleshooting and reproduction

- `uv: command not found`: install uv using its linked installation guide,
  then reopen the terminal so its executable is on `PATH`.
- Cannot find `examples/demo_prefix_cache.py`: run from the checkout root,
  where `pyproject.toml` and `examples/` are present.
- `ModuleNotFoundError: nano_lmcache`: use `uv sync --locked` and
  `uv run --locked ...` so the project environment is selected.
- Every new run starts cold: this demo uses process-local memory. Run the
  disk demo twice to explore persistence across processes.
- Hit rate differs from the first table: check the part heading; each part
  has its own cache and counters. Changing requests or chunk size also
  changes the number of eligible chunks.

The commands and excerpts above were checked on 2026-10-23 with Python
3.11.17, uv 0.13.5, Linux x86_64, and a fresh core-only environment without
PyTorch. They were run against this chapter's revision, based on upstream
`50b4b4a`. No timings or GPU results are reported in this chapter.
