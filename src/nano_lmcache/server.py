"""A small, localhost-oriented TCP server for a shared cache.

Each connection carries one request and response. A frame is a four-byte
big-endian length followed by a JSON header, a newline, and opaque payload
bytes. GET accepts an ordered list of keys and stops at the first miss.
This educational protocol has no authentication or encryption.
"""

import argparse
import json
import math
import socket
import socketserver
import struct
from collections.abc import Sequence
from typing import cast

from nano_lmcache.storage.memory import MemoryKVCache
from nano_lmcache.storage.protocol import CacheStorage
from nano_lmcache.storage.serializers import JsonSerializer, Serializer

MAX_FRAME_BYTES = 64 * 1024 * 1024
MAX_HEADER_BYTES = 64 * 1024


def _read_exact(connection: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise ConnectionError("incomplete frame")
        data.extend(chunk)
    return bytes(data)


def read_message(connection: socket.socket) -> tuple[dict[str, object], bytes]:
    """Read one bounded frame; TCP reads need not align with message boundaries."""
    (size,) = struct.unpack("!I", _read_exact(connection, 4))
    if not 0 < size <= MAX_FRAME_BYTES:
        raise ValueError("invalid frame length")
    header, separator, payload = _read_exact(connection, size).partition(b"\n")
    if not separator or len(header) > MAX_HEADER_BYTES:
        raise ValueError("invalid frame header")
    try:
        metadata = json.loads(header)
    except RecursionError as error:
        raise ValueError("frame header is too deeply nested") from error
    if not isinstance(metadata, dict):
        raise ValueError("header must be an object")
    return metadata, payload


def encode_message(
    metadata: dict[str, object], payloads: Sequence[bytes] = ()
) -> bytes:
    """Frame metadata and binary payloads without base64 or pickle."""
    header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    size = len(header) + 1 + sum(map(len, payloads))
    if len(header) > MAX_HEADER_BYTES or size > MAX_FRAME_BYTES:
        raise ValueError("message exceeds the frame limit")
    return struct.pack("!I", size) + header + b"\n" + b"".join(payloads)


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server = cast(CacheServer, self.server)
        self.request.settimeout(server.request_timeout)
        try:
            metadata, payload = read_message(self.request)
            if metadata.get("op") == "put":
                key, parent = metadata.get("key"), metadata.get("parent_key")
                if not isinstance(key, str) or not (
                    parent is None or isinstance(parent, str)
                ):
                    raise ValueError("invalid key or parent")
                value = server.serializer.loads(payload)
                if value is None:
                    raise ValueError("None is reserved for misses")
                server.storage.put(key, value, parent_key=parent)
                response = encode_message({"ok": True})
            elif metadata.get("op") == "get":
                keys = metadata.get("keys")
                if (
                    not isinstance(keys, list)
                    or any(not isinstance(key, str) for key in keys)
                    or payload
                ):
                    raise ValueError("invalid key list")
                payloads = []
                total_bytes = 0
                for key in keys:
                    value = server.storage.get(key)
                    if value is None:
                        break
                    data = server.serializer.dumps(value)
                    total_bytes += len(data)
                    if total_bytes > MAX_FRAME_BYTES:
                        raise ValueError("response exceeds the frame limit")
                    payloads.append(data)
                response = encode_message(
                    {"lengths": [len(data) for data in payloads]}, payloads
                )
            else:
                raise ValueError("unknown operation")
            self.request.sendall(response)
        except (OSError, ValueError, TypeError):
            # A malformed request or dropped connection must not stop the server.
            return


class CacheServer(socketserver.ThreadingTCPServer):
    """Serve a thread-safe CacheStorage; clients must use the same Serializer.

    The default address binds only localhost and chooses a free port. Call
    serve_forever() to run, shutdown() from another thread to stop, and
    server_close() (or use a context manager) to release the listening socket.
    """

    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 32

    def __init__(
        self,
        storage: CacheStorage,
        address: tuple[str, int] = ("127.0.0.1", 0),
        *,
        serializer: Serializer | None = None,
        timeout: float = 1.0,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.storage = storage
        self.serializer = serializer if serializer is not None else JsonSerializer()
        self.request_timeout = timeout
        super().__init__(address, _Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--max-entries", type=int, default=1024)
    args = parser.parse_args()
    with CacheServer(
        MemoryKVCache(max_entries=args.max_entries), ("127.0.0.1", args.port)
    ) as server:
        print(f"Cache server listening on {server.server_address}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
