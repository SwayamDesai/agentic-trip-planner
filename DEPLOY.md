# Deploying

## What makes this app awkward to deploy

Worth understanding before picking a platform, because three properties rule
most options out:

**1. Requests run for 60–150 seconds.** One plan is six agents against
free-tier models. Any platform with a function timeout shorter than that
(Vercel, Netlify, most serverless) cannot run it at all.

**2. It streams.** Progress arrives over SSE. A platform or CDN that buffers
responses turns the live agent panel into a two-minute blank screen. The app
sends `X-Accel-Buffering: no`, but a buffering proxy in front will still break it.

**3. It has real local state**, and losing it costs money rather than
convenience:

| Path | Holds | Cost of losing it |
|---|---|---|
| `/data/cache` | API responses, tiered 6h–365d | re-fetches, and re-spends metered SerpApi searches |
| `/data/trips.sqlite` | checkpoints | resumable trips become full re-runs, at ~40k tokens each |
| `/data/gateway.sqlite` | rate-limit buckets, API keys | every redeploy hands all callers a fresh quota |

The last one has a trap. The `limits` library has no file-backed store, so
without Redis it runs `memory://` — and then a volume persists nothing, because
nothing was on disk. Restarting the container took global credits from 594 back
to 600. The Dockerfile therefore sets `GATEWAY_LIMITER=bucket`, the
SQLite-backed implementation. With a Redis, switch back:
`GATEWAY_LIMITER=limits GATEWAY_STORAGE_URI=redis://…` — and that is the only
correct choice once there is more than one instance.

So: an ephemeral filesystem is not merely lossy here, it is a quota leak.

**4. Single process, deliberately.** Per-run metrics and the Langfuse trace id
are module-level, and the concurrency limiter is in-process. Multiple workers
would interleave them and each keep its own limiter — so N workers draw N times
the upstream quota while each believes it is compliant. Concurrency is bounded
by the gateway instead, which is what the token budget wants anyway.

---

## Fly.io (recommended)

Fits all four: persistent volumes on the free allowance, no request timeout,
and one always-addressable machine.

```bash
fly launch --no-deploy                    # reads fly.toml
fly volumes create atlas_data --size 1 --region ord

# secrets — never in the image or in fly.toml
fly secrets set \
  GROQ_API_KEY=... \
  GROQ_API_KEY_2=... \
  GROQ_API_KEY_3=... \
  SERPAPI_KEY=... \
  TRAVELPAYOUTS_TOKEN=... \
  GATEWAY_KEY_SALT="$(openssl rand -hex 16)"

fly deploy
fly logs
```

`GATEWAY_KEY_SALT` matters: API keys are stored as salted hashes, so changing
the salt invalidates every issued key. Set it once and leave it.

### After deploying

```bash
curl https://atlas-trip-planner.fly.dev/health
curl https://atlas-trip-planner.fly.dev/gateway/status
```

`/health` reports `degraded` rather than failing when no model key is set — a
container that boots without keys is running but cannot plan anything, and a
check that only proves the process started would call that healthy.

---

## Other platforms

| Platform | Verdict |
|---|---|
| **Render** | Works, but the free tier spins down after inactivity (~50s cold start) and has no persistent disk — so every wake starts with a cold cache |
| **Railway** | Fine technically; trial credits then paid |
| **Hugging Face Spaces** (Docker) | Good for a public demo, persistent storage on paid tiers only |
| **A small VPS** | Most control, and the state problem disappears. ~$5/mo |
| **Vercel / Netlify / Lambda** | Will not work — function timeouts are shorter than one plan, and there is no local disk |
| **Kubernetes** | Would work with a PVC and one replica, but a single-replica Deployment with a volume is most of what Fly gives you for far less setup |

---

## The thing to decide before making it public

A public deployment shares **your** upstream budget with every visitor. Three
Groq keys give ~600k tokens/day, which is roughly 15 fresh plans. One crawler
finding `/api/plan` drains that before you notice.

Mitigations, in order of how much they matter:

1. **The gateway, enabled** (`GATEWAY=1`, already the default in the Dockerfile).
   Anonymous callers get 40 credits/day — one plan. The global bucket caps the
   whole service at 600 credits/day regardless of how many callers there are.
2. **A demo mode** replaying a stored plan, so a visitor who arrives after the
   quota is spent still sees the system work rather than six 429s. Not built yet,
   and it is the single highest-value thing left.
3. **Issue keys** for anyone who needs real runs: `POST /gateway/keys`.

Without at least (1), do not put this on a public URL.

---

## Running the container locally first

```bash
docker build -t atlas .
docker run --rm -p 8000:8000 \
  -v atlas_data:/data \
  --env-file .env \
  atlas
```

Same image, same volume layout, so a problem shows up locally rather than in a
deploy. `--env-file .env` is fine locally; in production use the platform's
secret store so keys never enter the image or its history.
