"""A synchronous remote cache with optional prefix batching."""

import math
import socket
from collections.abc import Sequence

from nano_lmcache.server import encode_message, read_message
from nano_lmcache.storage.serializers import JsonSerializer, Serializer


class RemoteKVCache:
    """Share a CacheServer across clients, opening one connection per wire request.

    Network failures become misses or dropped writes. A failed write may
    already have reached the server, so writes are not retried. Timeout is
    a per-socket-operation limit, not a deadline for an entire prefix lookup.
    Client serializer errors propagate, as they do for DiskKVCache.
    """

    def __init__(
        self,
        address: tuple[str, int] = ("127.0.0.1", 5555),
        *,
        serializer: Serializer | None = None,
        timeout: float = 1.0,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self._address = address
        self._serializer = serializer if serializer is not None else JsonSerializer()
        self._timeout = timeout

    def _request(self, data: bytes) -> tuple[dict[str, object], bytes] | None:
        try:
            with socket.create_connection(self._address, self._timeout) as connection:
                connection.sendall(data)
                return read_message(connection)
        except (OSError, ValueError):
            return None

    def put(self, key: str, value: object, *, parent_key: str | None = None) -> None:
        """Send a payload and its parent; serialization errors remain visible."""
        if value is None:
            raise ValueError("None is reserved for misses")
        data = encode_message(
            {"op": "put", "key": key, "parent_key": parent_key},
            [self._serializer.dumps(value)],
        )
        self._request(data)

    def get(self, key: str) -> object | None:
        """Return a payload or None when absent or the server cannot be reached."""
        values = self.get_many([key])
        return values[0] if values else None

    def get_many(self, keys: Sequence[str]) -> list[object]:
        """Return the leading hits in order, stopping at the first missing key.

        Keys after a miss are never fetched or made more recent on the server.
        Key lists are split to fit the wire limits. A disconnected or malformed
        frame stops the lookup, retaining hits from earlier batches. Payload
        decoding errors remain visible so incompatible serializers can be diagnosed.
        """
        if not keys:
            return []
        try:
            data = encode_message({"op": "get", "keys": list(keys)})
        except ValueError:
            if len(keys) == 1:
                raise
            middle = len(keys) // 2
            leading = self.get_many(keys[:middle])
            if len(leading) < middle:
                return leading
            return leading + self.get_many(keys[middle:])
        response = self._request(data)
        if response is None:
            return []
        metadata, payload = response
        lengths = metadata.get("lengths")
        if (
            not isinstance(lengths, list)
            or len(lengths) > len(keys)
            or any(type(size) is not int or size < 0 for size in lengths)
            or sum(lengths) != len(payload)
        ):
            return []
        values: list[object] = []
        offset = 0
        for size in lengths:
            value = self._serializer.loads(payload[offset : offset + size])
            if value is None:
                return []
            values.append(value)
            offset += size
        return values
