# Running it, and serving it to users

## Two processes, not one

Since the async split there are two things to run:

```
api      serves the website, accepts jobs, reports results   — never plans
worker   takes jobs off the queue and plans them             — never serves HTTP
```

**The most common mistake is starting only the API.** Plans then sit in the
queue forever, which looks exactly like a hang. Check `GET /jobs/stats`: if
`queued` climbs and `running` stays 0, no worker is running.

---

## Development

```bash
./run.sh                       # starts both, reload on save
```

Or by hand, in two terminals:

```bash
.venv/bin/python worker.py
.venv/bin/python -m uvicorn api:app --reload --port 8000
```

Then open <http://127.0.0.1:8000>.

---

## Production-shaped, on one machine

```bash
docker compose up --build
docker compose up --scale worker=3      # three workers, same queue
```

Same image for both roles, so they cannot drift apart in dependencies.

Watch the split work:

```bash
curl localhost:8000/jobs/stats          # queue depth
docker compose logs -f worker           # jobs being claimed
```

`--scale worker=3` is the payoff: capacity now comes from adding workers, not
from making the web server bigger.

### On a Linux server without Docker

systemd, two units — `atlas-api.service` and `atlas-worker.service` — both with
`Restart=always`, and `TimeoutStopSec=200` on the worker so SIGTERM lets it
finish its current job. That timeout is the same concern as Docker's
`stop_grace_period`: too short and a restart kills work mid-plan, which is the
thing the queue exists to prevent.

---

## How a user actually reaches it

Three levels, increasing exposure.

### 1. Just you

```bash
./run.sh      # http://127.0.0.1:8000
```

### 2. Your network

`run.sh` already binds `0.0.0.0`, so anyone on the LAN can use
`http://<your-ip>:8000`. Fine for showing someone across the room.

### 3. The public internet, without cloud

**Cloudflare Tunnel** — no open ports, no port forwarding, works behind NAT,
free, and you get HTTPS:

```bash
cloudflared tunnel --url http://localhost:8000
```

**Tailscale Funnel** is the same idea if you already run Tailscale.

Port-forwarding your router works too, but then TLS, DNS and being directly
exposed are all yours to handle. A tunnel is strictly less work and less risk.

**Before you expose it publicly, read the quota section below.** That is not a
formality.

---

## What the user's browser actually does

```
1. GET  /                     the single-page app
2. POST /api/chat             one LLM call reads the message into fields
3. POST /plans                → 202 {job_id} in ~5ms
4. GET  /plans/{id}/events    SSE: agent progress, then the result
```

Steps 3 and 4 are the split: accepting the work and watching it are separate
requests, so nothing depends on one long-lived connection. A browser that
reconnects resumes with `?after=<last event id>` instead of replaying.

---

## Before exposing it publicly

A public URL shares **your** LLM budget with every visitor. Three Groq keys give
roughly 600k tokens a day — about 15 plans. One crawler finding `POST /plans`
drains that before you notice.

1. **Keep the gateway on** (`GATEWAY=1`, already the compose default).
   Anonymous callers get 40 credits a day: one plan. The global bucket caps the
   whole service regardless of how many callers there are.
2. **Set the spend cap** at the provider, if you have moved to a paid model.
   Rate limits protect the service; a spend cap protects you.
3. **Issue keys** to anyone who needs real volume: `POST /gateway/keys`.

---

## Checking it is healthy

```bash
curl localhost:8000/health          # process up, and whether keys are configured
curl localhost:8000/jobs/stats      # queue depth and oldest waiting job
curl localhost:8000/gateway/status  # credits left, requests in flight
```

`oldest_queued_age_seconds` is the one to watch. Queue *depth* cannot tell a
short burst from a stall; the age of the oldest waiting job can.
