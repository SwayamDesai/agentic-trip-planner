"""The prompt registry: Langfuse can serve prompts, but never break planning.

The tests are mostly about the failure directions, because those are the ones
that matter in production: an unreachable registry, a truncated publish, a
prompt that expects a variable the deployed code does not have. Every one of
them must degrade to the prompt shipped in the image.
"""

import re
from pathlib import Path

import pytest

from providers import prompts, tracing
from providers.prompt_defaults import DEFAULTS

ROOT = Path(__file__).resolve().parent.parent


class FakePrompt:
    def __init__(self, text, version=7):
        self.prompt = text
        self.version = version


class FakeClient:
    """Stands in for the Langfuse client, counting calls."""

    def __init__(self, text=None, raises=False, version=7):
        self.text = text
        self.raises = raises
        self.version = version
        self.calls = 0

    def get_prompt(self, name, **_kwargs):
        self.calls += 1
        if self.raises:
            raise RuntimeError("langfuse unreachable")
        return FakePrompt(self.text, self.version)


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    prompts.clear_cache()
    monkeypatch.setattr(prompts, "PROMPT_DIR", "")
    yield
    prompts.clear_cache()


def use(client, monkeypatch):
    monkeypatch.setattr(tracing, "client", lambda: client)
    return client


# --- the shipped defaults are the floor, so they must all be usable ---------


def test_every_default_renders_and_is_substantial():
    known = {"today": "2026-01-01", "min_per_day": 2, "max_per_day": 5}
    for name in prompts.names():
        text = prompts.get(name, **known).text
        assert len(text) > prompts.MIN_LENGTH
        assert "{{" not in text, f"{name} left a placeholder unfilled"


def test_every_prompt_the_code_asks_for_has_a_default():
    """A `prompts.get("x")` with no default would raise at runtime, in a node."""
    asked = set()
    for path in list((ROOT / "agents").glob("*.py")) + [ROOT / "chat.py"]:
        asked |= set(re.findall(r'prompts\.get\(\s*"(\w+)"', path.read_text()))
    assert asked, "no prompt lookups found — did the call sites move?"
    assert asked <= set(DEFAULTS), f"no default for {asked - set(DEFAULTS)}"


def test_unknown_prompt_is_a_programming_error():
    with pytest.raises(KeyError):
        prompts.get("does-not-exist")


# --- resolution order ------------------------------------------------------


def test_langfuse_version_wins_over_the_shipped_one(monkeypatch):
    published = "PUBLISHED PROMPT. " + "x" * 200
    use(FakeClient(published, version=12), monkeypatch)

    got = prompts.get("weather")
    assert got.text == published
    assert (got.source, got.version) == ("langfuse", 12)
    assert prompts.resolved("weather") == "langfuse:v12"


def test_unreachable_registry_falls_back_to_code(monkeypatch):
    client = use(FakeClient(raises=True), monkeypatch)

    got = prompts.get("weather")
    assert got.text == DEFAULTS["weather"]
    assert got.source == "code"
    assert client.calls == 1


def test_unconfigured_langfuse_costs_nothing(monkeypatch):
    monkeypatch.setattr(tracing, "client", lambda: None)
    assert prompts.get("flight").source == "code"


def test_local_directory_overrides_everything(monkeypatch, tmp_path):
    (tmp_path / "flight.md").write_text("LOCAL DRAFT. " + "y" * 200)
    monkeypatch.setattr(prompts, "PROMPT_DIR", str(tmp_path))
    use(FakeClient("PUBLISHED. " + "x" * 200), monkeypatch)

    got = prompts.get("flight")
    assert got.source == "file" and got.text.startswith("LOCAL DRAFT")


def test_local_directory_missing_file_falls_through(monkeypatch, tmp_path):
    monkeypatch.setattr(prompts, "PROMPT_DIR", str(tmp_path))
    use(FakeClient(raises=True), monkeypatch)
    assert prompts.get("flight").source == "code"


# --- a bad publish must not reach the model -------------------------------


def test_truncated_publish_is_rejected(monkeypatch):
    """Half a prompt is worse than an old prompt: it silently drops rules."""
    use(FakeClient("Return flights."), monkeypatch)

    got = prompts.get("flight")
    assert got.source == "code" and got.text == DEFAULTS["flight"]


def test_publish_wanting_an_unknown_variable_is_rejected(monkeypatch):
    """`{{currency}}` reaching the model reads as an instruction with a hole."""
    use(FakeClient("Report costs in {{currency}}. " + "x" * 200), monkeypatch)

    assert prompts.get("budget").source == "code"


def test_non_string_publish_is_rejected(monkeypatch):
    """A chat-type prompt is a list of messages; this registry serves text."""
    use(FakeClient([{"role": "system", "content": "x" * 200}]), monkeypatch)

    assert prompts.get("weather").source == "code"


# --- caching --------------------------------------------------------------


def test_a_fetched_prompt_is_cached(monkeypatch):
    client = use(FakeClient("PUBLISHED. " + "x" * 200), monkeypatch)

    for _ in range(5):
        prompts.get("weather")
    assert client.calls == 1


def test_a_failing_registry_is_dialled_once_not_once_per_agent(monkeypatch):
    """Otherwise a Langfuse outage adds its timeout to every node in the graph."""
    client = use(FakeClient(raises=True), monkeypatch)

    for _ in range(6):
        prompts.get("itinerary", min_per_day=2, max_per_day=5)
    assert client.calls == 1


def test_expired_cache_refetches(monkeypatch):
    client = use(FakeClient("PUBLISHED. " + "x" * 200), monkeypatch)
    prompts.get("weather")

    later = prompts.time.monotonic() + prompts.CACHE_TTL + 1
    monkeypatch.setattr(prompts.time, "monotonic", lambda: later)
    prompts.get("weather")
    assert client.calls == 2


def test_cache_is_per_prompt(monkeypatch):
    client = use(FakeClient("PUBLISHED. " + "x" * 200), monkeypatch)
    prompts.get("weather")
    prompts.get("flight")
    assert client.calls == 2


# --- variables come from code, not from prompt text ------------------------


def test_itinerary_density_numbers_come_from_the_code_constants(monkeypatch):
    """The scorer reads the same constants, so they cannot drift apart."""
    from agents.itinerary_agent import MAX_PER_DAY, MIN_PER_DAY

    use(
        FakeClient(
            "Plan days. {{min_per_day}} to {{max_per_day}} activities. " + "x" * 200
        ),
        monkeypatch,
    )
    text = prompts.get(
        "itinerary", min_per_day=MIN_PER_DAY, max_per_day=MAX_PER_DAY
    ).text
    assert f"{MIN_PER_DAY} to {MAX_PER_DAY} activities" in text


def test_chat_prompt_carries_todays_date():
    text = prompts.get("chat", today="2026-08-21").text
    assert "2026-08-21" in text


# --- seeding --------------------------------------------------------------


def test_push_does_not_overwrite_a_published_prompt(monkeypatch):
    """The registry is where prompts are edited; a deploy must not undo that."""

    class Existing(FakeClient):
        def create_prompt(self, **_kwargs):  # pragma: no cover - must not run
            raise AssertionError("push overwrote a published prompt")

    use(Existing("PUBLISHED. " + "x" * 200, version=3), monkeypatch)
    report = prompts.push()
    assert all("left alone" in line for line in report)


def test_push_creates_what_is_missing(monkeypatch):
    created = []

    class Empty(FakeClient):
        def get_prompt(self, name, **_kwargs):
            raise RuntimeError("not found")

        def create_prompt(self, name, prompt, **_kwargs):
            created.append((name, prompt))
            return FakePrompt(prompt, version=1)

    use(Empty(), monkeypatch)
    prompts.push()
    assert [n for n, _ in created] == [
        prompts.remote_name(n) for n in prompts.names()
    ]
    assert dict(created)[prompts.remote_name("flight")] == DEFAULTS["flight"]


def test_push_is_a_no_op_without_langfuse(monkeypatch):
    monkeypatch.setattr(tracing, "client", lambda: None)
    assert "not configured" in prompts.push()[0]
