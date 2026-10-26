"""Sharing, prefix semantics, and failure handling over real local sockets."""

import json
import socket
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Thread

import pytest

from nano_lmcache import server as wire
from nano_lmcache.core.prefix_cache import ChunkedPrefixCache
from nano_lmcache.server import MAX_FRAME_BYTES, read_message
from nano_lmcache.storage.disk import DiskKVCache
from nano_lmcache.storage.memory import MemoryKVCache
from nano_lmcache.storage.remote import RemoteKVCache
from nano_lmcache.storage.tiered import TieredStorage


class BytesSerializer:
    def dumps(self, value):
        if not isinstance(value, bytes):
            raise TypeError("expected bytes")
        return value

    def loads(self, data):
        return data


@pytest.mark.parametrize("backend", ["memory", "disk", "tiered"])
def test_two_clients_share_prefixes(serve_cache, tmp_path, backend):
    storage = MemoryKVCache()
    if backend == "disk":
        storage = DiskKVCache(tmp_path)
    elif backend == "tiered":
        storage = TieredStorage(fast=MemoryKVCache(), slow=DiskKVCache(tmp_path))
    server = serve_cache(storage)
    writer = ChunkedPrefixCache(
        chunk_size=2, namespace="shared", storage=RemoteKVCache(server.server_address)
    )
    reader = ChunkedPrefixCache(
        chunk_size=2, namespace="shared", storage=RemoteKVCache(server.server_address)
    )

    writer.store([1, 2, 3, 4, 5, 6], ["a", {"kv": [1, 2]}, "c"])
    result = reader.lookup([1, 2, 3, 4, 9, 9])

    assert result.values == ("a", {"kv": [1, 2]})
    parent_storage = storage.slow if isinstance(storage, TieredStorage) else storage
    assert parent_storage.parent_of(result.keys[1]) == result.keys[0]
    assert reader.metrics.hit_chunks == 2
    assert reader.metrics.miss_chunks == 1


def test_prefix_cache_uses_one_batched_request(serve_cache, monkeypatch):
    server = serve_cache(MemoryKVCache())
    remote = RemoteKVCache(server.server_address)
    cache = ChunkedPrefixCache(chunk_size=1, namespace="test", storage=remote)
    cache.store([1, 2, 3], ["a", "b", "c"])
    calls = []
    get_many = remote.get_many

    def record(keys):
        calls.append(keys)
        return get_many(keys)

    def no_single_get(key):
        pytest.fail("prefix lookup must use the batched operation")

    monkeypatch.setattr(remote, "get_many", record)
    monkeypatch.setattr(remote, "get", no_single_get)
    assert cache.lookup([1, 2, 3]).values == ("a", "b", "c")
    assert len(calls) == 1
    assert len(calls[0]) == 3


@pytest.mark.parametrize("available", [False, True])
def test_long_prefix_lookup_splits_requests(serve_cache, available):
    tokens = list(range(16_000))
    storage = MemoryKVCache(max_entries=1000)
    local = ChunkedPrefixCache(chunk_size=16, namespace="long", storage=storage)
    local.store(tokens, list(range(1000)))
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        address = (
            serve_cache(storage).server_address if available else reserved.getsockname()
        )
        remote = ChunkedPrefixCache(
            chunk_size=16, namespace="long", storage=RemoteKVCache(address, timeout=0.1)
        )
        result = remote.lookup(tokens)
    assert result.values == (tuple(range(1000)) if available else ())
    assert result.matched_tokens == (16_000 if available else 0)


@pytest.mark.parametrize("limit", ["header", "frame"])
def test_batches_measure_escaped_keys_at_exact_boundary(
    serve_cache, monkeypatch, limit
):
    keys = [f'key-{index}-\u00e9\\"\n' for index in range(5)]
    header = json.dumps({"op": "get", "keys": keys[:2]}, separators=(",", ":"))
    if limit == "header":
        monkeypatch.setattr(wire, "MAX_HEADER_BYTES", len(header.encode()))
    else:
        monkeypatch.setattr(wire, "MAX_FRAME_BYTES", len(header.encode()) + 1)
    storage = MemoryKVCache()
    for index, key in enumerate(keys):
        storage.put(key, index)
    remote = RemoteKVCache(serve_cache(storage).server_address)
    request = remote._request
    requests = []

    def record(data):
        requests.append(data)
        return request(data)

    monkeypatch.setattr(remote, "_request", record)
    assert remote.get_many(keys) == list(range(5))
    assert len(requests) == 3
    assert len(requests[0][4:]) == len(header.encode()) + 1


@pytest.mark.parametrize("header_limit", [64, wire.MAX_HEADER_BYTES])
def test_batch_stops_at_first_miss_without_touching_later_keys(
    serve_cache, monkeypatch, header_limit
):
    monkeypatch.setattr(wire, "MAX_HEADER_BYTES", header_limit)

    class TracedMemory(MemoryKVCache):
        def __init__(self):
            super().__init__(max_entries=2)
            self.reads = []

        def get(self, key):
            self.reads.append(key)
            return super().get(key)

    storage = TracedMemory()
    first, missing, later = [word * 4 for word in ["first", "missing", "later"]]
    storage.put(later, "later")
    storage.put(first, "first")
    remote = RemoteKVCache(serve_cache(storage).server_address)

    assert remote.get_many([first, missing, later]) == ["first"]
    assert storage.reads == [first, missing]
    storage.put("new", "new")
    assert later not in storage
    assert first in storage


def test_later_batch_failure_keeps_earlier_hits(monkeypatch):
    monkeypatch.setattr(wire, "MAX_HEADER_BYTES", 32)
    remote = RemoteKVCache()
    responses = iter([({"lengths": [1]}, b"1"), None])
    monkeypatch.setattr(remote, "_request", lambda data: next(responses))
    assert remote.get_many(["first", "second", "third"]) == [1]


def test_binary_serializer_round_trip_including_empty_payload(serve_cache):
    serializer = BytesSerializer()
    server = serve_cache(MemoryKVCache(), serializer=serializer)
    remote = RemoteKVCache(server.server_address, serializer=serializer)
    payloads = [b"", b"\x00\xff\n\x80"]
    for key, value in zip(["a", "b"], payloads, strict=True):
        remote.put(key, value)
    assert remote.get_many(["a", "b"]) == payloads
    assert remote.get("missing") is None


def test_client_serialization_failure_is_not_silently_dropped(serve_cache):
    remote = RemoteKVCache(serve_cache(MemoryKVCache()).server_address)
    with pytest.raises(TypeError):
        remote.put("key", object())
    with pytest.raises(ValueError, match="None"):
        remote.put("key", None)


@pytest.mark.parametrize(
    "response",
    [
        ({"lengths": [-1]}, b""),
        ({"lengths": [True]}, b"0"),
        ({"lengths": [3]}, b"0"),
        ({"lengths": [1, 1]}, b"00"),
        ({"lengths": "1"}, b"0"),
        ({"lengths": [4]}, b"null"),
    ],
)
def test_malformed_response_is_a_miss(monkeypatch, response):
    remote = RemoteKVCache()
    monkeypatch.setattr(remote, "_request", lambda data: response)
    assert remote.get("key") is None


def test_client_payload_decoding_errors_remain_visible(monkeypatch):
    remote = RemoteKVCache()
    monkeypatch.setattr(remote, "_request", lambda data: ({"lengths": [1]}, b"!"))
    with pytest.raises(ValueError):
        remote.get("key")


def test_concurrent_clients(serve_cache):
    storage = MemoryKVCache()
    server = serve_cache(storage)

    def worker(index):
        remote = RemoteKVCache(server.server_address)
        remote.put(f"root-{index}", index)
        remote.put(f"child-{index}", index + 1, parent_key=f"root-{index}")
        return remote.get_many([f"root-{index}", f"child-{index}"])

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, range(12)))
    assert results == [[index, index + 1] for index in range(12)]
    assert len(storage) == 24


def test_demo_shares_cache_between_separate_processes(serve_cache):
    server = serve_cache(MemoryKVCache())
    example = Path(__file__).resolve().parents[2] / "examples/demo_remote_cache.py"
    for extra in (["--store"], []):
        result = subprocess.run(
            [
                sys.executable,
                str(example),
                "--port",
                str(server.server_address[1]),
                *extra,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        assert "Reused 4/6 tokens" in result.stdout


def test_unreachable_server_is_a_miss_and_drops_writes():
    # Reserve a port without listening, avoiding a bind/close/rebind race.
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        remote = RemoteKVCache(reserved.getsockname(), timeout=0.1)
        assert remote.get("key") is None
        assert remote.get_many(["a", "b"]) == []
        assert remote.get_many([]) == []
        remote.put("key", "value")


def test_slow_server_times_out_as_a_miss(serve_cache):
    release = Event()

    class SlowMemory(MemoryKVCache):
        def get(self, key):
            release.wait(timeout=2)
            return super().get(key)

    remote = RemoteKVCache(serve_cache(SlowMemory()).server_address, timeout=0.01)
    try:
        assert remote.get("key") is None
    finally:
        release.set()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout_is_rejected(timeout):
    with pytest.raises(ValueError, match="timeout"):
        RemoteKVCache(("127.0.0.1", 1), timeout=timeout)


def test_fragmented_frame_is_read_completely():
    data = b'{"lengths":[4]}\n\x00\xff\n\x80'
    reader, writer = socket.socketpair()
    with reader, writer:
        reader.settimeout(1)

        def send():
            for byte in struct.pack("!I", len(data)) + data:
                writer.sendall(bytes([byte]))

        thread = Thread(target=send)
        thread.start()
        try:
            assert read_message(reader) == ({"lengths": [4]}, b"\x00\xff\n\x80")
        finally:
            thread.join()


@pytest.mark.parametrize(
    "data",
    [
        struct.pack("!I", MAX_FRAME_BYTES + 1),
        struct.pack("!I", 2) + b"[]",
        struct.pack("!I", 3) + b"[]\n",
        struct.pack("!I", 5) + b"bad\n!",
        struct.pack("!I", 20) + b"short",
        pytest.param(
            struct.pack("!I", 2001) + b"[" * 1000 + b"]" * 1000 + b"\n",
            id="deeply-nested-header",
        ),
    ],
)
def test_bad_frame_is_rejected(data):
    reader, writer = socket.socketpair()
    with reader, writer:
        reader.settimeout(1)
        writer.sendall(data)
        writer.shutdown(socket.SHUT_WR)
        with pytest.raises((ValueError, ConnectionError)):
            read_message(reader)


@pytest.mark.parametrize(
    "header",
    [
        b'{"op":"unknown"}',
        b'{"op":"get","keys":[1]}',
        b'{"op":"put","key":"a","parent_key":1}',
        pytest.param(
            b'{"op":' + b"[" * 1000 + b"]" * 1000 + b"}",
            id="deeply-nested-header",
        ),
    ],
)
def test_malformed_request_does_not_kill_server(serve_cache, header):
    server = serve_cache(MemoryKVCache())
    with socket.create_connection(server.server_address, timeout=1) as connection:
        data = header + b"\n"
        connection.sendall(struct.pack("!I", len(data)) + data)
        assert connection.recv(1) == b""
    remote = RemoteKVCache(server.server_address)
    remote.put("healthy", "value")
    assert remote.get("healthy") == "value"


def test_frame_limits_apply_to_requests_and_responses(serve_cache, monkeypatch):
    monkeypatch.setattr(wire, "MAX_FRAME_BYTES", 128)
    monkeypatch.setattr(wire, "MAX_HEADER_BYTES", 64)
    storage = MemoryKVCache()
    storage.put("large", "x" * 128)
    remote = RemoteKVCache(serve_cache(storage).server_address)
    assert remote.get("large") is None
    with pytest.raises(ValueError, match="limit"):
        remote.put("large", "x" * 128)
    with pytest.raises(ValueError, match="limit"):
        remote.get("key" * 64)
    with pytest.raises(ValueError, match="limit"):
        wire.encode_message({"header": "x" * 64})


def test_read_rejects_oversized_header(monkeypatch):
    monkeypatch.setattr(wire, "MAX_HEADER_BYTES", 4)
    data = b'{"op":"get"}\n'
    reader, writer = socket.socketpair()
    with reader, writer:
        reader.settimeout(1)
        writer.sendall(struct.pack("!I", len(data)) + data)
        with pytest.raises(ValueError, match="header"):
            read_message(reader)


def test_deep_response_header_is_a_miss(monkeypatch):
    reader, writer = socket.socketpair()
    with reader, writer:
        metadata = b'{"lengths":' + b"[" * 1000 + b"]" * 1000 + b"}\n"
        writer.sendall(struct.pack("!I", len(metadata)) + metadata)
        monkeypatch.setattr(socket, "create_connection", lambda *args: reader)
        assert RemoteKVCache().get("key") is None
