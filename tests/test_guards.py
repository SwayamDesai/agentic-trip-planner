"""Meta-tests: prove the suite's own safety rails work.

A guard nobody tests is a guard that silently stops working.
"""

import pytest
import requests


def test_network_is_blocked():
    """The autouse no_network fixture must actually intercept real calls."""
    with pytest.raises(AssertionError, match="real network access"):
        requests.get("https://example.com")


def test_cache_is_isolated(isolated_cache):
    """Tests must not read or write the developer's real cache dir."""
    from providers import cache

    assert cache.CACHE_DIR == isolated_cache
    assert "tmp" in str(cache.CACHE_DIR) or "pytest" in str(cache.CACHE_DIR)
