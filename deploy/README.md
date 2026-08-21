# Edge gateway

Two layers, because neither can do the other's job.

```
client
  │
  ├─ Kong (deploy/kong.yml) ......... edge: request-rate flood guard, body size,
  │                                   CORS, correlation id
  │
  └─ app (gateway/) ................. business metering: weighted credits,
                                      tiers, concurrency, refunds
```

## Why not put everything in Kong

Kong sees `POST /api/plan` and a header. It cannot know that request is ~40 LLM
calls and ~40k tokens while `POST /api/chat` is one call and ~1k. Counting both
as "one request" lets a caller spend 40x their share while looking compliant.

That is the same reason GitHub meters GraphQL in *points* computed per query,
and why OpenAI meters *tokens*: a request's cost is not derivable from its URL,
so the limit has to live where the cost is known.

## Why not do everything in the app

Kong is better at the coarse layer, and it is one config file instead of code:
flood protection, body limits, CORS, TLS termination, correlation ids, access
logs. All of it stateless, none of it needing to understand a trip.

## Running it

```bash
# 1. the app
./run.sh                                    # uvicorn on :8000

# 2. the edge (needs the Docker daemon running)
docker compose -f deploy/docker-compose.yml up

# 3. use the proxy, not the app directly
curl -i http://localhost:8080/api/chat -X POST \
     -H 'Content-Type: application/json' -d '{"message":"hi"}'
```

The response carries both layers' headers: `RateLimit-*` from Kong,
`X-RateLimit-Credits`/`X-Principal-Tier` from the app.

## Alternatives considered

| Tool | Why not here |
|---|---|
| **APISIX** | Equivalent features; needs etcd, so more moving parts for the same result |
| **Traefik** | Lightest, and a good choice if all you want is rate limiting — but fewer plugins if this grows |
| **Envoy** | The most capable global rate limiting, via a separate gRPC service. Real infrastructure, and far past what one process needs |
| **Tyk** | Strong on key management and quotas — which is precisely the part staying in the app |
| **AWS API Gateway / Cloudflare** | The right answer once deployed. Usage plans give burst+rate per key with no code at all |

## Fail-open, deliberately

`fault_tolerant: true` means Kong allows traffic through if its counter store is
unreachable. That is the right default for a rate limiter: the limiter is there
to protect against excess, not to be a single point of failure. A limiter that
fails *closed* takes the whole service down when its own storage hiccups.

The app's credit metering is the backstop, and it fails closed — so there is
still a limit even when the edge is degraded.
