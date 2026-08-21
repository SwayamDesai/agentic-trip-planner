"""A small API gateway, built to understand what a gateway does.

Not because this app needs one — a single-process planner on free tiers does
not. It is here because the concepts are worth knowing first-hand:

    identity.py      who is calling, and what limits apply
    buckets.py       token bucket rate limiting, weighted and persistent
    concurrency.py   in-flight caps, which rate limiting does not cover
    policy.py        what each route costs, in credits rather than requests
    errors.py        one refusal shape, with Retry-After
    middleware.py    the order the checks run in, and why
    admin.py         status and key management, deliberately unmetered

The one idea worth taking away: a gateway can cheaply enforce *how often*
someone calls you, but only the application knows what a call is *worth*. That
is why GitHub meters GraphQL in points and why OpenAI meters tokens — and why
the cost table lives here rather than in nginx.
"""

from gateway.middleware import GatewayMiddleware


def install(app, db_path: str = ".gateway.sqlite"):
    """Attach the gateway to a FastAPI app, and mount its admin routes.

    The stores are built here rather than inside the middleware, so the admin
    routes can share them without reaching into Starlette's middleware stack.
    """
    from gateway import admin, state

    state.build(db_path)
    app.add_middleware(GatewayMiddleware, db_path=db_path)
    app.include_router(admin.router)
    return app


__all__ = ["GatewayMiddleware", "install"]
