# Contributing

## Running the tests

```bash
python -m pytest          # 372 tests, ~4s, no network, no API keys
```

The suite is offline by design — an autouse fixture in `tests/conftest.py`
makes `requests.Session.request` raise. If a test needs HTTP, stub the specific
module attribute it uses (`places.requests.post`, `travel._serpapi`, …) rather
than relaxing the guard. A test that reaches a live API burns metered free-tier
quota and makes the suite too expensive to run often.

## Conventions worth knowing

**Anything derivable is derived, not generated.** Cost arithmetic, plan
rendering, nights, and the free/paid distinction for places are Python. If you
find yourself asking a model for a number that could be computed, compute it.

**Every prompt rule that matters has a check.** A rule that exists only in a
prompt is a request. Add the corresponding scorer in `evals/scorers.py` or a
check in `verify.py`.

**Findings are warnings, not corrections.** Name matching is fuzzy, so silently
deleting an "unverified" activity could remove a real place. Surface it.

**One agent, one state key.** That is what makes the parallel join safe without
merge logic. Shared keys (`errors`, `warnings`, `evidence`) carry append
reducers.

**Verify facts against providers, don't recall them.** Model names, API
endpoints and free-tier limits change. Check `/models`, read the 429 body, test
the call. Several APIs this project originally targeted no longer exist.

## Adding an agent

1. Module in `agents/`, using `run_agent` or `run_tool_agent`
2. Its own state key in `models.TripState`
3. A key chain in `providers/llm.AGENT_PROFILES` whose primary differs from any
   agent it runs concurrently with
4. A timeout in `agents/base.AGENT_TIMEOUTS`
5. A criticality tier in `status.REQUIRED` or `status.OPTIONAL`
6. A verifier, if its output makes checkable claims
