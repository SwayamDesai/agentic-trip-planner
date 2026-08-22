"""Prompt registry: Langfuse first, code second, never a broken prompt.

Prompts are the part of this system most likely to change and least likely to
need a deploy. Moving them into Langfuse means a wording fix is a publish, with
a version number and an author, and a trace can say which version produced a
given plan. What it must not mean is that an edit in a web UI can take planning
down.

So resolution is a chain, and every step can fail without consequence:

    local directory   PROMPT_DIR, if set. Development only: edit a file, rerun.
    Langfuse          the `production` label, cached for CACHE_TTL seconds.
    code              providers/prompt_defaults.py, which ships in the image.

Whatever comes back is VALIDATED before it is used — non-empty, long enough to
be a real prompt, and with every `{{placeholder}}` filled. A prompt that fails
validation is discarded and the next source is tried, so publishing a truncated
prompt, or one referring to a variable the code does not supply, degrades to the
shipped version instead of sending a mangled system message to the model.

Variables are filled from code, never from prompt text. The itinerary prompt is
the reason: it states an activities-per-day range that the density scorer also
enforces, so the numbers come from the same constants the scorer reads and a
prompt edit cannot make the instruction and the grader disagree.
"""

import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from providers.prompt_defaults import DEFAULTS

# Which published version the app follows. A prompt is edited freely in the UI;
# nothing reaches production until it carries this label.
LABEL = os.getenv("PROMPT_LABEL", "production")

# How long a fetched prompt is trusted before refetching. Long enough that a
# run makes at most one registry call per prompt, short enough that a publish
# takes effect without a restart.
CACHE_TTL = int(os.getenv("PROMPT_CACHE_TTL", "300"))

# Optional local override, for editing a prompt without publishing it. Absent
# in production; when present it wins, and traces say so.
PROMPT_DIR = os.getenv("PROMPT_DIR", "").strip()

# Prompts are namespaced in Langfuse so they do not collide with anything else
# in the project.
NAMESPACE = os.getenv("PROMPT_NAMESPACE", "atlas")

# A real prompt for this system is a few hundred characters. Anything shorter is
# a truncated paste or an empty draft, not an instruction.
MIN_LENGTH = 80

_PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")

_lock = threading.Lock()
_cache: dict[str, tuple[float, str, str, Optional[int]]] = {}

# Last resolution per prompt name, so the tracing layer can label a generation
# with the version that produced it without threading it through every agent.
_resolved: dict[str, tuple[str, Optional[int]]] = {}


@dataclass(frozen=True)
class Prompt:
    """A resolved, rendered prompt plus where it came from."""

    name: str
    text: str
    source: str  # "langfuse" | "file" | "code"
    version: Optional[int]

    @property
    def label(self) -> str:
        return f"{self.source}:v{self.version}" if self.version else self.source


def remote_name(name: str) -> str:
    return f"{NAMESPACE}/{name}"


def names() -> list[str]:
    return sorted(DEFAULTS)


def get(name: str, **variables) -> Prompt:
    """Resolve one prompt, render it, and record what was used.

    Never raises and never returns an unusable prompt: the shipped default is
    the floor, and it is validated on the same terms as a fetched one.
    """
    if name not in DEFAULTS:
        raise KeyError(
            f"unknown prompt {name!r}; add it to providers/prompt_defaults.py"
        )

    for source, raw, version in _candidates(name):
        rendered = _render(raw, variables)
        if rendered is None:
            continue
        with _lock:
            _resolved[name] = (source, version)
        return Prompt(name=name, text=rendered, source=source, version=version)

    # Unreachable in practice: the code default is validated by a test. Kept as
    # a raise rather than a silent empty string, because an agent with no system
    # prompt would produce confident nonsense instead of failing.
    raise RuntimeError(f"no usable prompt for {name!r}")


def resolved(name: str) -> Optional[str]:
    """Label for the version last served, for trace metadata. None if unused."""
    with _lock:
        entry = _resolved.get(name)
    if entry is None:
        return None
    source, version = entry
    return f"{source}:v{version}" if version else source


def _candidates(name: str):
    """Yield (source, raw text, version) in priority order."""
    if PROMPT_DIR:
        text = _from_dir(name)
        if text is not None:
            yield "file", text, None

    text, version = _from_langfuse(name)
    if text is not None:
        yield "langfuse", text, version

    yield "code", DEFAULTS[name], None


def _from_dir(name: str) -> Optional[str]:
    try:
        path = Path(PROMPT_DIR) / f"{name}.md"
        return path.read_text() if path.is_file() else None
    except OSError:
        return None


def _from_langfuse(name: str) -> tuple[Optional[str], Optional[int]]:
    """Fetch from the registry, through a TTL cache. Failure returns (None, None).

    The cache is checked before the client is even built, so an unconfigured or
    unreachable Langfuse costs one attempt per TTL window rather than one per
    agent call.
    """
    now = time.monotonic()
    with _lock:
        hit = _cache.get(name)
        if hit and hit[0] > now:
            return hit[1], hit[3]

    from providers import tracing

    client = tracing.client()
    if client is None:
        return None, None

    try:
        prompt = client.get_prompt(
            remote_name(name), label=LABEL, cache_ttl_seconds=CACHE_TTL
        )
        text = prompt.prompt
        version = getattr(prompt, "version", None)
    except Exception:  # noqa: BLE001 - an unreachable registry is not an outage
        # Negative result cached too. Without this, a Langfuse that is down
        # would be dialled once per agent per run, adding its timeout to every
        # node on the critical path.
        with _lock:
            _cache[name] = (now + CACHE_TTL, "", "miss", None)
        return None, None

    if not isinstance(text, str):
        return None, None

    with _lock:
        _cache[name] = (now + CACHE_TTL, text, "langfuse", version)
    return text, version


def _render(raw: str, variables: dict) -> Optional[str]:
    """Fill placeholders and validate, or None if the text is not usable.

    Returning None rather than raising is what makes the chain work: a bad
    version in the registry falls through to the shipped one instead of failing
    the agent that asked for it.
    """
    if not isinstance(raw, str) or len(raw.strip()) < MIN_LENGTH:
        return None

    filled = _PLACEHOLDER.sub(
        lambda m: str(variables[m.group(1)])
        if m.group(1) in variables
        else m.group(0),
        raw,
    )
    # A leftover placeholder means the published prompt expects a variable this
    # version of the code does not supply. Sending `{{max_per_day}}` to the
    # model reads as an instruction with a hole in it, so this text is rejected.
    if _PLACEHOLDER.search(filled):
        return None
    return filled


def clear_cache() -> None:
    """Drop cached prompts. For tests, and for a manual reload."""
    with _lock:
        _cache.clear()
        _resolved.clear()


def push(dry_run: bool = False) -> list[str]:
    """Seed the registry from the shipped defaults.

    Creates each prompt at the `production` label if it is not there already.
    Deliberately does NOT overwrite an existing one: the registry is the place
    prompts are edited, and a deploy that stamped the code version back over a
    published fix would make the UI a lie.
    """
    from providers import tracing

    client = tracing.client()
    if client is None:
        return ["langfuse is not configured; nothing pushed"]

    report: list[str] = []
    for name in names():
        remote = remote_name(name)
        try:
            existing = client.get_prompt(remote, label=LABEL, cache_ttl_seconds=0)
            report.append(f"{remote}: exists at v{existing.version}, left alone")
            continue
        except Exception:  # noqa: BLE001 - not found is the normal path here
            pass
        if dry_run:
            report.append(f"{remote}: would create")
            continue
        try:
            created = client.create_prompt(
                name=remote,
                prompt=DEFAULTS[name],
                labels=[LABEL],
                type="text",
            )
            report.append(f"{remote}: created v{getattr(created, 'version', '?')}")
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            report.append(f"{remote}: FAILED {type(exc).__name__}: {exc}")
    return report


def status() -> list[str]:
    """What each prompt currently resolves to, and from where."""
    lines = []
    for name in names():
        # variables are irrelevant to provenance, so ask for the raw resolution
        for source, raw, version in _candidates(name):
            mark = "" if version is None else f" v{version}"
            lines.append(f"{name:10} {source}{mark}  {len(raw)} chars")
            break
    return lines


if __name__ == "__main__":  # pragma: no cover - operator entry point
    import sys

    argv = sys.argv[1:]
    if "status" in argv:
        print("\n".join(status()))
    else:
        print("\n".join(push(dry_run="--dry-run" in argv)))
