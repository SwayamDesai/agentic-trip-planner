"""What a run cost, and refusing to guess when it cannot be known.

The subtle part is not the arithmetic — it is that the app asks the proxy for a
routing alias (`planner`) and the response body echoes that alias back. Pricing
the alias would report every proxied call as unpriced; pricing it as zero would
report every run as free. The served deployment only appears in a response
header, so these tests pin the extraction as much as the sums.
"""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from providers import metrics, pricing
from providers.prices import RATES

MODEL = "groq/openai/gpt-oss-120b"


# --- rate lookup ----------------------------------------------------------


def test_the_generated_table_is_not_empty():
    """A table that silently emptied would price every run at zero."""
    assert len(RATES) >= 5
    assert all(
        isinstance(v, tuple) and len(v) == 2 and v[0] >= 0 and v[1] >= 0
        for v in RATES.values()
    )


def test_the_same_model_prices_the_same_under_every_naming():
    """The proxy, the provider and Langfuse each name this model differently."""
    expected = pricing.rate(MODEL)
    assert expected is not None
    assert pricing.rate("openai/gpt-oss-120b") == expected
    # a bare name resolves only while it is unambiguous in the table
    assert pricing.rate("gpt-oss-120b") == expected


def test_an_ambiguous_bare_name_is_refused(monkeypatch):
    """Two providers can charge different rates for the same model name."""
    monkeypatch.setitem(pricing.RATES, "provider-a/thing", (1e-06, 2e-06))
    monkeypatch.setitem(pricing.RATES, "provider-b/thing", (9e-06, 9e-06))
    assert pricing.rate("thing") is None


@pytest.mark.parametrize("alias", ["planner", "planner-small"])
def test_a_routing_alias_has_no_price(alias):
    """An alias is a group of deployments, not a model. Pricing it invents a rate."""
    assert pricing.rate(alias) is None


@pytest.mark.parametrize("value", [None, "", "   ", "no/such/model"])
def test_unknown_models_are_unpriced_not_free(value):
    assert pricing.rate(value) is None
    assert pricing.cost(value, 1000, 1000) is None


def test_cost_is_input_and_output_priced_separately():
    per_in, per_out = pricing.rate(MODEL)
    assert per_in != per_out, "the test is meaningless if the rates are equal"
    assert pricing.cost(MODEL, 1000, 0) == pytest.approx(1000 * per_in)
    assert pricing.cost(MODEL, 0, 1000) == pytest.approx(1000 * per_out)
    assert pricing.cost(MODEL, 1000, 500) == pytest.approx(
        1000 * per_in + 500 * per_out
    )


# --- reading the response -------------------------------------------------


def _result(*, body_model=None, headers=None):
    message = AIMessage(content="ok", response_metadata={"headers": headers or {}})
    return LLMResult(
        generations=[[ChatGeneration(message=message)]],
        llm_output={
            "model_name": body_model,
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 50},
        },
    )


def test_the_served_deployment_beats_the_alias_in_the_body():
    result = _result(
        body_model="planner", headers={"x-litellm-model-name": MODEL}
    )
    assert metrics._model_of(result) == MODEL


def test_without_headers_the_body_model_is_used():
    assert metrics._model_of(_result(body_model=MODEL)) == MODEL


def test_the_proxy_reported_cost_is_read():
    result = _result(headers={"x-litellm-response-cost": "1.335e-05"})
    assert metrics._reported_cost(result) == pytest.approx(1.335e-05)


@pytest.mark.parametrize("raw", ["", "not-a-number", None])
def test_an_unparseable_cost_header_is_ignored(raw):
    result = _result(headers={"x-litellm-response-cost": raw})
    assert metrics._reported_cost(result) is None


def test_headers_are_matched_case_insensitively():
    result = _result(headers={"X-LiteLLM-Response-Cost": "0.5"})
    assert metrics._reported_cost(result) == 0.5


# --- recording ------------------------------------------------------------


def test_the_proxy_figure_wins_over_the_local_table():
    """One number per call, from the layer that actually billed it."""
    run_id = metrics.new_run()
    metrics.record_llm("weather", 100, 50, MODEL, reported_cost=0.0025)
    run = metrics.finish(run_id)
    assert run.agents["weather"].cost_usd == 0.0025
    assert run.agents["weather"].unpriced_calls == 0


def test_a_direct_call_falls_back_to_the_table():
    run_id = metrics.new_run()
    metrics.record_llm("weather", 1000, 500, MODEL)
    run = metrics.finish(run_id)
    assert run.agents["weather"].cost_usd == pytest.approx(
        pricing.cost(MODEL, 1000, 500)
    )


def test_an_unpriceable_call_is_counted_so_the_total_reads_as_a_floor():
    run_id = metrics.new_run()
    metrics.record_llm("weather", 1000, 500, "planner")
    run = metrics.finish(run_id)
    totals = run.totals()
    assert totals["unpriced_calls"] == 1
    assert totals["cost_usd"] == 0.0
    assert totals["total_tokens"] == 1500  # tokens are still counted


def test_cost_totals_sum_across_agents():
    run_id = metrics.new_run()
    metrics.record_llm("weather", 1000, 500, MODEL)
    metrics.record_llm("flight", 2000, 800, MODEL)
    run = metrics.finish(run_id)
    expected = pricing.cost(MODEL, 3000, 1300)
    assert run.totals()["cost_usd"] == pytest.approx(round(expected, 6))


def test_cost_appears_per_agent_in_the_payload_the_ui_reads():
    run_id = metrics.new_run()
    metrics.record_llm("flight", 1000, 500, MODEL)
    payload = metrics.finish(run_id).as_dict()
    assert payload["agents"]["flight"]["cost_usd"] > 0
    assert "cost_usd" in payload["totals"]
    assert "unpriced_calls" in payload["totals"]
