"""Keep the core pipeline regression tests offline."""

import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import socket

    def deny(*args, **kwargs):
        raise AssertionError("network forbidden")

    for name in ["connect", "connect_ex", "bind"]:
        monkeypatch.setattr(socket.socket, name, deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)
