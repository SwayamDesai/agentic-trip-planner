# Deploying to Hugging Face Spaces

> **No longer free.** Docker Spaces now require a PRO subscription — only
> Static Spaces are free, and a static host cannot run this. Kept because the
> configuration is correct if you have PRO.
>
> **For a free always-on deployment, use [../oracle/SETUP.md](../oracle/SETUP.md).**

Uses the repository's `Dockerfile` unchanged.

## Why a Space fits this project

* **One container is enough.** The browser uses the single-request `/api/plan`
  path, so no separate worker process is needed.
* **No request timeout** that would cut off a 150s plan.
* **Secrets UI**, so keys never enter the image or git.

The tradeoff: **storage is ephemeral** on the free tier. The cache, checkpoints
and rate-limit buckets reset whenever the Space restarts, so a cold start
re-spends some quota rebuilding the cache. Acceptable for a demo; it is why the
writable paths below point at `/tmp`.

## Steps

**1. Create the Space**

At https://huggingface.co/new-space — SDK **Docker**, template **Blank**,
visibility **Public**.

**2. Push the code**

The Space is a git repo. Its `README.md` must carry the YAML frontmatter that
configures the Space, so the copy in this directory replaces the project one:

```bash
git clone https://huggingface.co/spaces/<you>/atlas-trip-planner hf-space
cd hf-space

# everything except the project README
rsync -a --exclude '.git' --exclude 'README.md' \
      --exclude '.venv' --exclude '.cache' --exclude '*.sqlite' \
      --exclude 'evals/fixtures' \
      ../ ./

cp deploy/hf-space/README.md ./README.md    # the frontmatter one

git add -A && git commit -m "Deploy Atlas" && git push
```

**3. Set secrets**

Space → **Settings → Variables and secrets**. As **secrets**:

| Secret | Needed for |
|---|---|
| `GROQ_API_KEY` | all agents |
| `GROQ_API_KEY_2`, `GROQ_API_KEY_3` | spreads the daily token cap |
| `SERPAPI_KEY` | hotel rates, flight fallback |
| `TRAVELPAYOUTS_TOKEN` | optional flight fallback |
| `GATEWAY_KEY_SALT` | any random string; keep it stable |

As plain **variables** (not secrets), because storage is ephemeral:

```
TRIP_CACHE_DIR = /tmp/cache
TRIP_DB        = /tmp/trips.sqlite
GATEWAY_DB     = /tmp/gateway.sqlite
JOBS_DB        = /tmp/jobs.sqlite
GATEWAY        = 1
GATEWAY_LIMITER = bucket
```

`/data` is only writable with paid persistent storage; `/tmp` always is.

**4. Check it**

```
https://<you>-atlas-trip-planner.hf.space/health
https://<you>-atlas-trip-planner.hf.space/
```

`/health` reports `degraded` if no model key is set — a container that boots
without keys is running but cannot plan anything.

## Keeping the gateway on

Traffic will be low, but a public URL still meets crawlers. `GATEWAY=1` caps
anonymous callers and the service as a whole, so one automated loop cannot
drain the day's budget before a recruiter opens the link.

## If you would rather have a real server

A €4/month Hetzner VPS removes every caveat above — persistent disk, no cold
start, no ephemeral storage. `docker compose up -d` plus Caddy for TLS. Worth it
if the link matters.
