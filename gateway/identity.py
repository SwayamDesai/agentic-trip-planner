"""Who is calling.

Every other gateway concern needs an answer to this, because a limit has to be
scoped to something. Three common schemes, in increasing strength:

  IP address      free, no setup, and nearly useless: everyone behind one NAT
                  shares a limit, and anyone can change it. Still worth having
                  as the fallback, or an unauthenticated endpoint has no scope
                  at all and one script can drain a shared quota.
  API key         a bearer secret in a header. Stable, scriptable, revocable.
  session/JWT     better for browsers, and carries claims, but needs an auth
                  system this project does not have.

API keys here, IP as the fallback. Keys are stored HASHED: the gateway needs to
recognise a key, not to be able to print one, and a leaked store should not be
a leaked credential list.
"""

import hashlib
import hmac
import ipaddress
import os
import secrets
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Tiers exist so limits are data, not code. Adding a paid tier should be a row.
TIERS = {
    "anonymous": {"daily_credits": 40, "burst": 40, "max_concurrent": 1},
    "free": {"daily_credits": 200, "burst": 80, "max_concurrent": 1},
    "pro": {"daily_credits": 2000, "burst": 400, "max_concurrent": 3},
}


@dataclass(frozen=True)
class Principal:
    """The identified caller, and the limits that apply to it."""

    id: str                 # bucket key, e.g. "key:3f2a" or "ip:203.0.113.7"
    tier: str
    label: str = ""         # human-readable, for logs
    authenticated: bool = False

    @property
    def limits(self) -> dict:
        return TIERS.get(self.tier, TIERS["anonymous"])


def _hash(raw: str) -> str:
    """Key digest. Salted from the environment so the store is not a rainbow
    table of short keys."""
    salt = os.getenv("GATEWAY_KEY_SALT", "atlas-gateway")
    return hashlib.sha256(f"{salt}:{raw}".encode()).hexdigest()


class KeyStore:
    def __init__(self, path: Path | str):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_hash TEXT PRIMARY KEY,
                    prefix   TEXT NOT NULL,
                    tier     TEXT NOT NULL,
                    label    TEXT NOT NULL DEFAULT '',
                    revoked  INTEGER NOT NULL DEFAULT 0
                )
                """
            )

    def issue(self, tier: str = "free", label: str = "") -> str:
        """Mint a key. Returned once — only its hash is stored."""
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(TIERS)}")
        raw = "atl_" + secrets.token_urlsafe(24)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO api_keys(key_hash, prefix, tier, label) VALUES(?,?,?,?)",
                (_hash(raw), raw[:8], tier, label),
            )
        return raw

    def resolve(self, raw: Optional[str]) -> Optional[Principal]:
        """Identify a key, or None if absent, unknown or revoked."""
        if not raw:
            return None
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT prefix, tier, label, revoked FROM api_keys WHERE key_hash = ?",
                (_hash(raw),),
            ).fetchone()
        if row is None or row[3]:
            return None
        prefix, tier, label, _ = row
        # the bucket key is derived from the hash, so the raw secret never
        # reaches a log line or a metric label
        return Principal(
            id=f"key:{_hash(raw)[:16]}",
            tier=tier,
            label=label or prefix,
            authenticated=True,
        )

    def revoke(self, raw: str) -> bool:
        with sqlite3.connect(self.path) as conn:
            cur = conn.execute(
                "UPDATE api_keys SET revoked = 1 WHERE key_hash = ?", (_hash(raw),)
            )
        return bool(cur.rowcount)

    def list_keys(self) -> list[dict]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT prefix, tier, label, revoked FROM api_keys ORDER BY prefix"
            ).fetchall()
        return [
            {"prefix": p, "tier": t, "label": l, "revoked": bool(r)}
            for p, t, l, r in rows
        ]


def extract_key(headers) -> Optional[str]:
    """Pull a key from the request.

    Both forms are accepted because both are common: `Authorization: Bearer …`
    is the convention, `X-API-Key` is what people reach for with curl.
    """
    auth = headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return headers.get("x-api-key") or None


def anonymous(client_ip: str) -> Principal:
    """Fallback identity, so unauthenticated traffic still has a scope."""
    return Principal(id=f"ip:{client_ip}", tier="anonymous", label=client_ip)


# Proxies whose X-Forwarded-For may be believed. Empty by default: trusting the
# header unconditionally lets any caller mint a fresh identity per request and
# walk straight past the anonymous limit, which is worse than having no limit,
# because it looks like one is enforced.
#
# In this deployment the app is not published to the host — only Caddy can reach
# it — so the compose file sets this to the private ranges Docker assigns.
def _trusted_networks() -> list:
    raw = os.getenv("GATEWAY_TRUSTED_PROXIES", "").strip()
    networks = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            networks.append(ipaddress.ip_network(item, strict=False))
        except ValueError:
            continue        # a typo must not become "trust everything"
    return networks


def _is_trusted(address: str, networks: list) -> bool:
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    return any(parsed in network for network in networks)


def client_ip(peer: Optional[str], forwarded_for: Optional[str] = None) -> str:
    """The caller's address, seen through however many proxies front the app.

    Behind Caddy, `request.client.host` is CADDY — so every anonymous visitor
    on the internet shared one bucket, and the whole deploy served one plan a
    day. Reading X-Forwarded-For fixes that, but only carefully:

      * the header is believed only when the immediate peer is a trusted proxy,
        otherwise a caller supplies whatever address it likes;
      * the chain is walked from the RIGHT, skipping trusted hops, because a
        client can prepend entries to the left but cannot stop the proxy from
        appending its own view of who connected.
    """
    if not peer:
        return "unknown"
    networks = _trusted_networks()
    if not networks or not _is_trusted(peer, networks) or not forwarded_for:
        return peer

    for candidate in reversed([p.strip() for p in forwarded_for.split(",")]):
        if candidate and not _is_trusted(candidate, networks):
            try:
                ipaddress.ip_address(candidate)
            except ValueError:
                continue    # garbage in the chain is ignored, not trusted
            return candidate
    return peer


def verify(raw: str, expected_hash: str) -> bool:
    """Constant-time comparison, so a timing signal cannot leak a key."""
    return hmac.compare_digest(_hash(raw), expected_hash)
