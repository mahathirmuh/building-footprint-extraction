"""Suite-wide guard: tests must run offline (CLAUDE.md), so any non-loopback internet connection fails loudly.

The guard raises RuntimeError rather than OSError, so libraries cannot quietly fall back to a local cache
(e.g. Hugging Face Hub weights); tests use ``encoder_weights=None`` and local fixtures instead.
"""

import socket

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}
_INET = (socket.AF_INET, socket.AF_INET6)
_REAL_CONNECT = socket.socket.connect


def _guarded_connect(sock, address, *args, **kwargs):
    if sock.family in _INET and address[0] not in _LOOPBACK:
        raise RuntimeError(f"network access is disabled during tests (attempted {address[0]!r})")
    return _REAL_CONNECT(sock, address, *args, **kwargs)


def pytest_configure(config):
    socket.socket.connect = _guarded_connect


def pytest_unconfigure(config):
    del socket.socket.connect
