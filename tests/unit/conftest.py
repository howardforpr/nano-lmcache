"""Local TCP servers shared by remote-storage tests."""

from contextlib import ExitStack
from threading import Thread

import pytest

from nano_lmcache.server import CacheServer


@pytest.fixture
def serve_cache():
    with ExitStack() as stack:

        def serve(storage, **kwargs):
            server = stack.enter_context(CacheServer(storage, **kwargs))
            thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
            thread.start()
            stack.callback(thread.join)
            stack.callback(server.shutdown)
            return server

        yield serve
