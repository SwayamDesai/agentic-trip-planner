"""Per-run metrics and traces.

A single module-level collector worked only while one plan ran at a time.
Workers break that, and the failure is silent: the second run's `reset()`
discarded the first run's counters mid-flight. Measured before the change —
ten LLM calls across two concurrent plans, seven recorded.
"""

import threading
import time

import pytest

from providers import metrics


@pytest.fixture(autouse=True)
def clean_registry():
    yield
    for run_id in list(metrics._RUNS):
        metrics.finish(run_id)


# --- isolation ---


def test_concurrent_runs_do_not_share_counters():
    results = {}

    def plan(name, calls):
        run_id = metrics.new_run()
        for _ in range(calls):
            metrics.record_llm(name, 1000, 100)
            time.sleep(0.01)
        results[name] = metrics.finish(run_id)

    a = threading.Thread(target=plan, args=("planA", 5))
    b = threading.Thread(target=plan, args=("planB", 5))
    a.start()
    time.sleep(0.02)          # overlap them deliberately
    b.start()
    a.join()
    b.join()

    for name in ("planA", "planB"):
        assert results[name].agents[name].llm_calls == 5, (
            f"{name} lost calls to the other run"
        )


def test_each_run_gets_a_distinct_id():
    a, b = metrics.new_run(), metrics.new_run()
    assert a != b


def test_finish_removes_the_collector():
    """A long-lived worker would otherwise accumulate one per plan it ran."""
    run_id = metrics.new_run()
    assert metrics.get(run_id) is not None
    metrics.finish(run_id)
    assert metrics.get(run_id) is None


def test_finishing_twice_is_harmless():
    run_id = metrics.new_run()
    assert metrics.finish(run_id) is not None
    assert metrics.finish(run_id) is None


# --- the thread-boundary problem, which is why the id also lives in state ---


def test_worker_thread_records_nothing_without_binding():
    """LangGraph runs each fan-out agent in a thread it created. Context does
    not cross that boundary, so a node must bind itself from the state."""
    run_id = metrics.new_run()
    unbound = []

    def agent_thread():
        # no bind_run: this thread has no idea which run it serves
        metrics.record_llm("flight", 1000, 100)
        unbound.append(metrics._CURRENT_RUN.get())

    t = threading.Thread(target=agent_thread)
    t.start()
    t.join()

    assert unbound == [None], "context leaked across the thread boundary"
    assert metrics.get(run_id).agents == {}, "recording went nowhere, as expected"


def test_binding_from_state_reconnects_the_thread():
    run_id = metrics.new_run()

    def agent_thread():
        metrics.bind_run({"run_id": run_id})     # what every node does first
        metrics.record_llm("flight", 1000, 100)

    t = threading.Thread(target=agent_thread)
    t.start()
    t.join()

    assert metrics.get(run_id).agents["flight"].llm_calls == 1


def test_bind_run_tolerates_missing_state():
    metrics.bind_run(None)
    metrics.bind_run({})
    metrics.record_llm("x", 1, 1)      # must not raise


def test_recording_outside_a_run_is_dropped_silently():
    """Scripts and tests call these without a run. Better to drop than raise."""
    metrics._CURRENT_RUN.set(None)
    metrics.record_llm("x", 1, 1)
    metrics.record_tool("x")
    metrics.record_retry("x")
    metrics.record_outcome("x", "done", 1.0)


# --- every node binds ---


def test_all_node_entry_points_bind():
    """A node that forgets to bind records into the void, which is invisible."""
    import inspect

    from agents import base, budget_agent, places_agent

    for fn in (base.run_agent, base.run_tool_agent,
               places_agent.places_agent, budget_agent.budget_agent):
        source = inspect.getsource(fn)
        assert "bind(state)" in source, f"{fn.__name__} does not bind"


def test_state_schema_carries_the_run_id():
    from typing import get_type_hints

    from models import TripState

    assert "run_id" in get_type_hints(TripState, include_extras=True)


# --- tracing is scoped the same way ---


def test_trace_ids_are_keyed_by_run():
    from providers import tracing

    a, b = metrics.new_run(), metrics.new_run()
    tracing._trace_ids[a] = "trace-a"
    tracing._trace_ids[b] = "trace-b"

    metrics._CURRENT_RUN.set(a)
    assert tracing._trace_id() == "trace-a"
    metrics._CURRENT_RUN.set(b)
    assert tracing._trace_id() == "trace-b"

    tracing._trace_ids.clear()


def test_trace_id_is_none_outside_a_run():
    from providers import tracing

    metrics._CURRENT_RUN.set(None)
    assert tracing._trace_id() is None
