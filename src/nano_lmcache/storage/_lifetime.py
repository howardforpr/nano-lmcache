"""Keep removed payloads alive until the outer storage operation unlocks."""

from collections.abc import Iterator
from contextlib import contextmanager
from threading import local


class _Payloads(local):
    retired: list[object] | None = None


_payloads = _Payloads()


@contextmanager
def defer_release() -> Iterator[list[object]]:
    retired = _payloads.retired
    if retired is not None:
        yield retired
        return

    retired = []
    _payloads.retired = retired
    try:
        yield retired
    finally:
        # Reset first: a finalizer can start another complete storage operation.
        _payloads.retired = None
        retired.clear()
