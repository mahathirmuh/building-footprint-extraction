"""The conftest guard really blocks internet access, so "tests need no internet" is enforced, not hoped for."""

import socket

import pytest


def test_internet_connection_is_blocked():
    with socket.socket() as sock, pytest.raises(RuntimeError, match="network access is disabled"):
        sock.connect(("192.0.2.1", 80))  # TEST-NET-1 literal address: no DNS lookup involved
