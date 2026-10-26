"""Share chunks between separate processes through a local cache server.

First run: uv run python -m nano_lmcache.server
Then run:  uv run python examples/demo_remote_cache.py --store
Then run:  uv run python examples/demo_remote_cache.py

Long prompts are split into bounded read batches, stopping at the first miss.
"""

import argparse

from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.storage.remote import RemoteKVCache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--store", action="store_true")
    args = parser.parse_args()
    cache = ChunkedPrefixCache(
        chunk_size=2,
        namespace="shared-demo",
        storage=RemoteKVCache(("127.0.0.1", args.port)),
    )
    if args.store:
        cache.store([1, 2, 3, 4, 5, 6], ["shared-0", "shared-1", "tail-a"])
    result = cache.lookup([1, 2, 3, 4, 9, 10])
    print(f"Reused {result.matched_tokens}/6 tokens: {result.values}")
    print("Stop the server and run again: an unavailable cache returns zero hits.")


if __name__ == "__main__":
    main()
