"""Who the gateway thinks is calling, once a proxy is in front of it.

This is the failure that only appears in production. Locally the caller IS the
peer and everything works; behind Caddy, `request.client.host` is Caddy, so
every anonymous visitor on the internet shares one 40-credit bucket and the
whole deploy serves one plan a day.

The fix has a sharp edge: believing X-Forwarded-For unconditionally is WORSE
than not reading it, because then any caller mints a fresh identity per request
and walks past the limit while the dashboard still shows one being enforced. So
these tests are mostly about refusing to believe the header.
"""

import pytest

from gateway.identity import client_ip

PRIVATE = "172.16.0.0/12,10.0.0.0/8,192.168.0.0/16,127.0.0.1/32"


@pytest.fixture
def trusted(monkeypatch):
    monkeypatch.setenv("GATEWAY_TRUSTED_PROXIES", PRIVATE)


@pytest.fixture
def untrusted(monkeypatch):
    monkeypatch.delenv("GATEWAY_TRUSTED_PROXIES", raising=False)


# --- refusing to believe the header ---------------------------------------


def test_without_configured_proxies_the_header_is_ignored(untrusted):
    """The default is to trust nothing, so a fresh deploy cannot be spoofed."""
    assert client_ip("203.0.113.9", "1.2.3.4") == "203.0.113.9"


def test_a_direct_caller_cannot_forge_an_identity(trusted):
    """The peer is a public address, so it is the client, whatever it claims."""
    assert client_ip("203.0.113.9", "10.9.9.9, 1.2.3.4") == "203.0.113.9"


def test_a_client_cannot_prepend_its_way_to_a_new_bucket(trusted):
    """Everything left of the proxy's own entry is caller-supplied text.

    Reading the leftmost entry — the common mistake — would let one script use a
    different identity per request.
    """
    forged = "9.9.9.9, 8.8.8.8, 198.51.100.23"
    assert client_ip("172.25.0.7", forged) == "198.51.100.23"


def test_garbage_in_the_chain_is_skipped_not_trusted(trusted):
    assert client_ip("172.25.0.7", "not-an-ip, 198.51.100.23") == "198.51.100.23"


def test_a_malformed_trusted_proxies_setting_trusts_nothing(monkeypatch):
    """A typo must not read as "trust everything"."""
    monkeypatch.setenv("GATEWAY_TRUSTED_PROXIES", "172.16.0.0/notacidr")
    assert client_ip("172.25.0.7", "198.51.100.23") == "172.25.0.7"


# --- believing it when it is safe to ---------------------------------------


def test_behind_a_trusted_proxy_the_real_client_is_used(trusted):
    assert client_ip("172.25.0.7", "198.51.100.23") == "198.51.100.23"


def test_two_visitors_behind_the_proxy_get_different_buckets(trusted):
    """The whole point: one visitor must not spend everyone's allowance."""
    first = client_ip("172.25.0.7", "198.51.100.23")
    second = client_ip("172.25.0.7", "203.0.113.77")
    assert first != second


def test_a_chain_of_trusted_hops_resolves_past_all_of_them(trusted):
    """Caddy behind a load balancer: skip every hop that is itself trusted."""
    chain = "198.51.100.23, 10.0.0.5, 172.25.0.7"
    assert client_ip("172.25.0.7", chain) == "198.51.100.23"


def test_ipv6_clients_resolve(trusted):
    assert client_ip("172.25.0.7", "2001:db8::42") == "2001:db8::42"


def test_no_header_falls_back_to_the_peer(trusted):
    assert client_ip("172.25.0.7", None) == "172.25.0.7"


def test_an_all_trusted_chain_falls_back_to_the_peer(trusted):
    """Nothing in the chain identifies a client, so do not invent one."""
    assert client_ip("172.25.0.7", "10.0.0.5, 172.25.0.7") == "172.25.0.7"


def test_a_missing_peer_is_named_rather_than_crashing(trusted):
    assert client_ip(None, "198.51.100.23") == "unknown"
