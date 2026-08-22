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

## First boot: things to do once

These are one-time, and skipping them is not fatal — the app runs without any of
them, which is exactly why they are easy to forget.

```bash
make up-full          # or `make up-llm` on a small instance; see the sizing note
make prompts-push     # seed the Langfuse prompt registry from the shipped text
make prompts          # confirm: every prompt should now read `langfuse v1`
make prices           # regenerate the price table from the proxy's own map
make health
```

Without `prompts-push`, prompts resolve to the copies compiled into the image.
Planning works identically — the point of the fallback chain — but nothing is
editable from the Langfuse UI, and the plan page reports `code` instead of
`langfuse:v1`. Run `make prompts` once to see which you are on.

### Sizing

The full stack includes self-hosted Langfuse, which brings ClickHouse, MinIO and
a second Postgres database. That needs the ARM shape (4 OCPU / 24 GB). On the
x86 micro shape (1 OCPU / 1 GB) ClickHouse alone will evict the app: run
`make up-llm` and point `LANGFUSE_HOST` at Langfuse Cloud instead, which has a
free tier.

### The daily allowance

The gateway grants 600 credits a day and a plan costs 40, so the deploy serves
about 15 plans a day. That is deliberate — it is roughly what the free Groq
quota (200k tokens per key per day) sustains, and it stops one visitor from
spending the week's allowance in an afternoon. A refused request answers 429
with `Retry-After`, and the page turns that into "today's allowance is spent, it
refills in about N hours" rather than an error code. Raise it in
`gateway/state.py` (`GLOBAL_DAILY_CREDITS`) if the model quota behind it grows.

## Oracle Cloud Always Free (recommended)

A real always-on VM, free permanently, with persistent disk. Step-by-step in
[deploy/oracle/SETUP.md](deploy/oracle/SETUP.md) — including the iptables trap
that makes an Oracle VM look unreachable even after the console firewall is open.

## Fly.io

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
