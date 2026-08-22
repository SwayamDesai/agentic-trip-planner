"""Report which mandatory settings are still missing from .env.

The compose files mark required values with `${VAR:?...}`, which fails the whole
stack at start-up. That is the right behaviour and a poor first experience: the
error names one variable, you fill it in, and the next start names the next one.
This lists all of them at once, before anything is started.

Requirements are read FROM the compose files rather than duplicated here, so a
new mandatory setting cannot be forgotten in two places.
"""

import pathlib
import re

COMPOSE_DIR = pathlib.Path(__file__).resolve().parent.parent / "deploy" / "oracle"
ENV = pathlib.Path(__file__).resolve().parent.parent / ".env"


def required() -> list[str]:
    text = "".join(p.read_text() for p in sorted(COMPOSE_DIR.glob("*.yml")))
    return sorted(set(re.findall(r"\$\{([A-Z_][A-Z0-9_]*):\?", text)))


def present() -> dict[str, str]:
    if not ENV.exists():
        return {}
    values = {}
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def main() -> int:
    env = present()
    if not ENV.exists():
        print(".env does not exist yet — copy .env.example to .env first")
        return 1

    missing = [key for key in required() if not env.get(key)]
    if not missing:
        print(f"all {len(required())} mandatory settings present")
        return 0

    print("missing from .env:")
    for key in missing:
        print(f"  {key}")
    print("\nGenerate secrets with: openssl rand -hex 32")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
