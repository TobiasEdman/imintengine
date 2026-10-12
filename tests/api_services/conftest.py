"""Scoped offline guard. All monkeypatches are restored after each test."""
import socket
import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Service regression attempted network access")
    for name in ("connect", "connect_ex", "bind"):
        monkeypatch.setattr(socket.socket, name, denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
