# Shared cache and batching

Run a server in one terminal, then a writer and a reader in another:

```bash
uv run python -m nano_lmcache.server --port 5555
uv run python examples/demo_remote_cache.py --store
uv run python examples/demo_remote_cache.py
```

Both client invocations print `Reused 4/6 tokens` because the second prompt
shares its first two chunks. Stop the server and repeat the reader: it
returns zero hits. Cache availability does not determine whether a caller
can compute its request.

The server can wrap memory, disk, or tiered storage. For a persistent
server, construct `CacheServer(DiskKVCache(directory))` and call
`serve_forever()`. Keep that disk directory owned by the server process.
Clients and server must use compatible serializers; JSON is the default,
and a custom serializer's bytes are transferred unchanged in the frame.
Client serialization errors remain visible; network and framing failures
become misses or dropped writes. Long key lists are split into smaller batches
to fit the request limits. Reads stop at the first miss or failed batch and
retain the leading hits from completed batches.

## Measured round-trip cost

```bash
uv run python benchmarks/bench_remote_cache.py \
    --chunks 1 16 64 128 --payload-bytes 4096 --repeats 25
```

Measured on 2026-10-23 using Python 3.11.17 on Linux x86-64, Intel Xeon
Platinum 8352Y (2.20 GHz), through TCP loopback. The server runs in a
separate thread of the benchmark process. Both modes use the same
fully-hit chain and 4096-character ASCII string per chunk, encoded with
JSON. Every measured lookup verifies its hit count. The table reports
median whole-prefix latency after three warm-ups; modes alternate order.

| Chunks | Per-chunk requests (ms) | Batched request (ms) | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.425 | 0.409 | 1.04x |
| 16 | 6.145 | 0.981 | 6.27x |
| 64 | 26.305 | 3.206 | 8.20x |
| 128 | 54.497 | 6.113 | 8.91x |

The baseline exposes only `put` and `get`, so prefix lookup sends one
request per chunk. The batched mode exposes `get_many`, sending one key
list and receiving the leading hits in a single response. Both include
key hashing, serialization, connection setup, and server thread creation.
The gain therefore includes fewer connections as well as fewer round
trips; this is not a measurement against a pooled-connection client.
These small JSON payloads do not predict tensor-transfer throughput or
performance across hosts. Re-run on the target machine with representative
payload sizes. Requests are split to fit the 64 KiB header and 64 MiB frame
limits; a batch whose response exceeds either limit becomes a miss. A single
oversized key or put raises `ValueError` locally.

## Relation to LMCache

LMCache's [remote backend](https://github.com/LMCache/LMCache/blob/ff9185d/lmcache/v1/storage_backend/remote_backend.py)
separates remote access from storage, supports batched retrieval, and
turns failed or timed-out requests into cache misses. This implementation
uses the same ideas with a synchronous standard-library client and server.
It keeps nano-lmcache's prefix-safe parent links and stops reading at the
first miss. It does not reproduce LMCache's asynchronous workers,
connector plugins, memory allocation, or transfer machinery, and the
wire protocols are not compatible.
