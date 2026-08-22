"""Turning token counts into a dollar figure.

Separate from `prices.py` so the generated table stays a table: `make prices`
overwrites that file wholesale, and nothing worth reviewing should live in a
file a script rewrites.

The interesting part is name resolution. The app asks the proxy for `planner`,
an alias covering three deployments, and the proxy answers with whichever it
routed to — so the model that must be priced is the one in the RESPONSE, and it
comes back in shapes that differ per provider and per client:

    groq/openai/gpt-oss-120b     the proxy's deployment id
    openai/gpt-oss-120b          the same model, as the provider names it
    planner                      the alias, which is not a model and has no price

The first two must price identically or the same run costs different amounts
depending on which layer reported it. The third must price as None, not zero.
"""

from typing import Optional

from providers.prices import RATES

# Aliases this project defines in the proxy config. They are routing labels, not
# models, so they have no rate — and must not silently borrow one.
ALIASES = frozenset({"planner", "planner-small"})


def rate(model: Optional[str]) -> Optional[tuple[float, float]]:
    """(input, output) USD per token for a model id, or None if unknown.

    Tries the id as given, then with each provider prefix stripped, then with
    the providers this project uses prepended — so `openai/gpt-oss-120b` finds
    the `groq/openai/gpt-oss-120b` entry it is the same model as.
    """
    if not model:
        return None
    name = model.strip()
    if not name or name in ALIASES:
        return None

    candidates = [name]
    if "/" in name:
        # groq/openai/gpt-oss-120b -> openai/gpt-oss-120b -> gpt-oss-120b
        parts = name.split("/")
        candidates += ["/".join(parts[i:]) for i in range(1, len(parts))]
    candidates += [f"groq/{name}", f"openrouter/{name}"]

    for candidate in candidates:
        found = RATES.get(candidate)
        if found:
            return found
    return _by_suffix(name)


def _by_suffix(name: str) -> Optional[tuple[float, float]]:
    """Resolve a bare model name, but only when exactly one entry can match.

    `gpt-oss-120b` is unambiguous today and resolves. A name two providers
    both list is refused rather than resolved to whichever sorted first: they
    can charge different rates, and a plausible wrong price is worse than a
    visible unpriced call.
    """
    tail = "/" + name
    matches = {
        rates for model, rates in RATES.items() if model.endswith(tail)
    }
    return matches.pop() if len(matches) == 1 else None


def cost(model: Optional[str], prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    """List-price cost of one call, or None when the model has no known rate."""
    rates = rate(model)
    if rates is None:
        return None
    per_input, per_output = rates
    return prompt_tokens * per_input + completion_tokens * per_output


def priced_models() -> int:
    return len(RATES)
