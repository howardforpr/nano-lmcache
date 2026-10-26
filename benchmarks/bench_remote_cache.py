"""Compare per-chunk and batched prefix lookups against the same local server.

    uv run python benchmarks/bench_remote_cache.py --chunks 1 16 64 128

Both modes use fresh TCP connections per request and the same JSON payloads.
Report median whole-prefix latency, including connections and serialization.
"""

import argparse
import platform
from statistics import median
from threading import Thread
from time import perf_counter

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.server import CacheServer
from nano_lmcache.storage.memory import MemoryKVCache
from nano_lmcache.storage.remote import RemoteKVCache


class PerChunkStorage:
    """Expose only the original two-method contract for the baseline."""

    def __init__(self, remote: RemoteKVCache) -> None:
        self.remote = remote

    def put(self, key: str, value: object, *, parent_key: str | None = None) -> None:
        self.remote.put(key, value, parent_key=parent_key)

    def get(self, key: str) -> object | None:
        return self.remote.get(key)


def run_scenario(
    remote: RemoteKVCache, chunks: int, payload_bytes: int, repeats: int
) -> tuple[float, float]:
    """Return per-chunk and batched median seconds; verify every lookup's hits."""
    tokens = list(range(chunks))
    batched = ChunkedPrefixCache(chunk_size=1, namespace="bench", storage=remote)
    single = ChunkedPrefixCache(
        chunk_size=1, namespace="bench", storage=PerChunkStorage(remote)
    )
    batched.store(tokens, ["x" * payload_bytes] * chunks)
    samples: list[list[float]] = [[], []]
    for repeat in range(repeats + 3):
        # Alternate order to reduce systematic warm-up and scheduling bias.
        order = (0, 1) if repeat % 2 == 0 else (1, 0)
        for mode in order:
            start = perf_counter()
            result = (single, batched)[mode].lookup(tokens)
            elapsed = perf_counter() - start
            assert result.matched_chunks == chunks, result
            if repeat >= 3:
                samples[mode].append(elapsed)
    return median(samples[0]), median(samples[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", nargs="+", type=int, default=[1, 16, 64, 128])
    parser.add_argument("--payload-bytes", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=25)
    args = parser.parse_args()
    if min(args.chunks) < 1 or args.payload_bytes < 0 or args.repeats < 1:
        parser.error("chunks/repeats must be positive and payload bytes nonnegative")
    print(f"Python {platform.python_version()}, {platform.platform()}")
    print(f"Payload: {args.payload_bytes} bytes; repeats: {args.repeats}; median ms")
    print("chunks | per-chunk | batched | speedup")
    with CacheServer(MemoryKVCache()) as server:
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            remote = RemoteKVCache(server.server_address, timeout=5)
            for chunks in args.chunks:
                single, batched = run_scenario(
                    remote, chunks, args.payload_bytes, args.repeats
                )
                print(
                    f"{chunks:6d} | {single * 1000:9.3f} | "
                    f"{batched * 1000:7.3f} | {single / batched:6.2f}x"
                )
        finally:
            server.shutdown()
            thread.join()


if __name__ == "__main__":
    main()
