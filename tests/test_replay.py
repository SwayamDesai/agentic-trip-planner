"""Record/replay and baseline comparison.

The property that matters: replay must never silently fall through to a live
call. A replay that quietly went live would be neither free nor reproducible,
while appearing to be both.
"""

import json
from pathlib import Path

import pytest

from evals import baseline as baseline_mod
from providers import replay


@pytest.fixture(autouse=True)
def temp_fixtures(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "FIXTURES", tmp_path / "fixtures")
    monkeypatch.delenv("EVAL_MODE", raising=False)


# --- modes ---


def test_live_is_the_default():
    assert replay.mode() == "live"
    assert not replay.active()


@pytest.mark.parametrize("value,expected", [("record", True), ("replay", True), ("live", False)])
def test_active_only_for_record_and_replay(monkeypatch, value, expected):
    monkeypatch.setenv("EVAL_MODE", value)
    assert replay.active() is expected


# --- keys reflect everything that changes the answer ---


def test_same_call_same_key():
    msgs = [{"role": "user", "content": "hello"}]
    assert replay.key_for("weather", msgs, 0.1, None) == replay.key_for(
        "weather", msgs, 0.1, None
    )


@pytest.mark.parametrize("mutate", [
    lambda a: {**a, "agent": "flight"},
    lambda a: {**a, "temperature": 0.9},
    lambda a: {**a, "messages": [{"role": "user", "content": "different"}]},
])
def test_key_changes_with_inputs(mutate):
    args = {"agent": "weather", "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.1, "schema": None}
    base = replay.key_for(args["agent"], args["messages"], args["temperature"], None)
    changed = mutate(args)
    assert replay.key_for(
        changed["agent"], changed["messages"], changed["temperature"], None
    ) != base


def test_prompt_edit_misses_on_purpose():
    """A changed prompt SHOULD miss — that is the signal you want when checking
    whether an edit helped."""
    before = replay.key_for("itinerary", [{"role": "system", "content": "old rules"}], 0.4, None)
    after = replay.key_for("itinerary", [{"role": "system", "content": "new rules"}], 0.4, None)
    assert before != after


def test_schema_is_part_of_the_key():
    from models import ItineraryResult, WeatherResult

    msgs = [{"role": "user", "content": "same"}]
    assert replay.key_for("x", msgs, 0.1, WeatherResult) != replay.key_for(
        "x", msgs, 0.1, ItineraryResult
    )


# --- round trip ---


def test_pydantic_round_trip():
    from models import TripScope

    value = TripScope(nights=4, reasoning="a capital worth four nights")
    replay.store("scope", "k1", value)
    restored = replay.lookup("scope", "k1", TripScope)
    assert restored.nights == 4
    assert restored.reasoning == value.reasoning


def test_message_round_trip():
    from langchain_core.messages import AIMessage

    msg = AIMessage(content="", tool_calls=[
        {"name": "get_weather", "args": {"lat": 1.0}, "id": "c1", "type": "tool_call"}
    ])
    replay.store("weather", "k2", msg)
    restored = replay.lookup("weather", "k2", None)
    assert restored.tool_calls[0]["name"] == "get_weather"


def test_missing_recording_raises_rather_than_going_live():
    with pytest.raises(replay.ReplayMiss, match="Re-record"):
        replay.lookup("weather", "nonexistent", None)


def test_unwritable_recording_does_not_fail_the_run(monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", boom)
    replay.store("weather", "k3", {"a": 1})   # must not raise


def test_stats_counts_fixtures():
    replay.store("weather", "a", {"x": 1})
    replay.store("weather", "b", {"x": 2})
    replay.store("flight", "c", {"x": 3})
    s = replay.stats()
    assert s["total"] == 3 and s["agents"]["weather"] == 2


# --- baseline comparison ---


def test_no_baseline_is_not_a_failure():
    out = baseline_mod.compare({"quality": {"groundedness": 1.0}}, None)
    assert out["status"] == "no_baseline" and out["regressions"] == []


def test_regression_is_detected():
    before = {"quality": {"groundedness": 1.0}}
    after = {"quality": {"groundedness": 0.7}}
    out = baseline_mod.compare(after, before)
    assert out["status"] == "regressed"
    assert out["regressions"][0]["delta"] == -0.3


def test_noise_below_tolerance_is_not_a_regression():
    """Models are non-deterministic; a regression must exceed the wobble."""
    before = {"quality": {"groundedness": 1.0}}
    after = {"quality": {"groundedness": 1.0 - baseline_mod.TOLERANCE / 2}}
    assert baseline_mod.compare(after, before)["status"] == "ok"


def test_improvement_is_reported_separately():
    out = baseline_mod.compare(
        {"quality": {"density": 0.9}}, {"quality": {"density": 0.5}}
    )
    assert out["status"] == "ok" and out["improvements"][0]["delta"] == 0.4


def test_a_scorer_that_stopped_running_is_a_regression():
    """A check that silently disappeared is worse than one that scored badly."""
    out = baseline_mod.compare({"quality": {}}, {"quality": {"groundedness": 1.0}})
    assert out["status"] == "regressed"
    assert "no longer runs" in out["regressions"][0]["detail"]


def test_new_scorers_are_noted_not_penalised():
    out = baseline_mod.compare(
        {"quality": {"groundedness": 1.0, "brand_new": 0.5}},
        {"quality": {"groundedness": 1.0}},
    )
    assert out["status"] == "ok" and out["new_scorers"] == ["brand_new"]


def test_baseline_save_and_load_round_trip(tmp_path):
    path = tmp_path / "baseline.json"
    baseline_mod.save({"quality": {"groundedness": 0.9}, "reliability": 0.8}, path)
    loaded = baseline_mod.load(path)
    assert loaded["quality"]["groundedness"] == 0.9
    assert loaded["reliability"] == 0.8


def test_corrupt_baseline_is_treated_as_absent(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text("{not json")
    assert baseline_mod.load(path) is None
