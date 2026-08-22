# Hosting on Oracle Cloud Always Free

A real always-on VM, free permanently. No cold start, persistent disk, and a
public URL you can put on a CV.

Oracle asks for a card to verify identity. It is not charged as long as you stay
on Always Free shapes — but the account can be upgraded accidentally, so pick
the shapes named below and nothing else.

---

## 1. Create the VM

Sign up at [cloud.oracle.com](https://cloud.oracle.com), then
**Compute → Instances → Create instance**:

| Field | Value | Why |
|---|---|---|
| Image | **Ubuntu 22.04** or 24.04 | The provision script is written for apt |
| Shape | **VM.Standard.A1.Flex** | The Always Free ARM shape |
| OCPUs / memory | **2 OCPU / 12 GB** | Always Free covers 4 OCPU / 24 GB total; half leaves room for a second VM later |
| Boot volume | 50 GB | Within the free 200 GB |
| SSH key | upload your public key | `cat ~/.ssh/id_ed25519.pub` |

**Check the shape says "Always Free-eligible".** If it does not, you have picked
a billable shape.

> **If you get "Out of host capacity"** — A1 capacity is genuinely scarce in
> popular regions. Try a different availability domain, or a different region
> when creating the account. Retrying over a few hours usually works; the
> shape is free precisely because it is oversubscribed.

## 2. Open the ports in the VCN

**Networking → Virtual Cloud Networks → your VCN → the public subnet → its
security list → Add ingress rules:**

| Source | Protocol | Port |
|---|---|---|
| `0.0.0.0/0` | TCP | 80 |
| `0.0.0.0/0` | TCP | 443 |

**This alone is not enough** — see the next step.

## 3. Provision

```bash
ssh ubuntu@<your-public-ip>

curl -fsSL https://raw.githubusercontent.com/SwayamDesai/agentic-trip-planner/main/deploy/oracle/provision.sh | bash
```

That installs Docker, clones the repo, adds swap, and **opens 80/443 in the
instance's own iptables**.

> That last part is the classic Oracle trap. Ubuntu images on Oracle ship
> iptables rules that drop everything except SSH, so opening the VCN security
> list in the console is only half the job. A VM that answers SSH but nothing
> else is almost always this.

## 4. HTTPS with a free domain

Let's Encrypt will not issue a certificate for a bare IP, so you need a name.
[DuckDNS](https://www.duckdns.org) gives you one free in about two minutes:
sign in, pick a subdomain, and point it at your VM's public IP.

Then in `~/atlas/.env`:

```
DOMAIN=atlas-yourname.duckdns.org
```

Caddy obtains and renews the certificate itself — no certbot, no cron job.
Leave `DOMAIN` unset and it serves plain HTTP instead.

## 5. Secrets and start

```bash
nano ~/atlas/.env
```

```
GROQ_API_KEY=...
GROQ_API_KEY_2=...
GROQ_API_KEY_3=...
SERPAPI_KEY=...
TRAVELPAYOUTS_TOKEN=...
GATEWAY_KEY_SALT=<any stable random string>
DOMAIN=atlas-yourname.duckdns.org
```

`GATEWAY_KEY_SALT` must stay stable: API keys are stored as salted hashes, so
changing it invalidates every key you have issued.

```bash
cd ~/atlas/deploy/oracle
docker compose up -d --build
```

First build takes a few minutes on ARM. Then:

```bash
curl localhost/health
docker compose logs -f
```

Visit `https://atlas-yourname.duckdns.org`.

---

## Adding the LLM gateway

```bash
cd ~/atlas
make up-data     # postgres + redis
make up-llm      # + litellm
```

Extra `.env` entries (generate each with `openssl rand -hex 24`):

```
POSTGRES_PASSWORD=...
LITELLM_MASTER_KEY=sk-...
LITELLM_SALT_KEY=...
LITELLM_DATABASE_URL=postgresql://atlas:<password>@postgres:5432/litellm
REDIS_URL=redis://redis:6379/0
LITELLM_BASE_URL=http://litellm:4000
```

`LITELLM_BASE_URL` is the switch. Set, and LLM traffic routes through the proxy,
which distributes across the three Groq keys and records spend. Unset, and the
app talks to the providers directly exactly as before.

Spend, per model:

```bash
docker compose exec postgres psql -U atlas -d litellm -c \
  'SELECT model, sum(spend), count(*) FROM "LiteLLM_SpendLogs" GROUP BY model'
```

Note that LiteLLM writes spend logs asynchronously — a row appears seconds
after the request, not immediately.

## Operating it

```bash
cd ~/atlas/deploy/oracle

docker compose logs -f app          # follow the app
docker compose ps                   # what is running
docker compose restart app          # restart
git -C ~/atlas pull && docker compose up -d --build    # deploy an update
docker compose exec app python -m pytest -q            # tests, on the server
```

**Survives reboot.** Docker is enabled at boot and the containers are
`restart: unless-stopped`, so an Oracle maintenance reboot brings the app back
with no action from you.

**State lives in a volume.** `atlas_data` holds the cache, checkpoints and
rate-limit counters, so a rebuild does not throw away work that cost metered API
calls. Certificates live in `caddy_data` — keep it, or Caddy re-issues on every
deploy and can hit Let's Encrypt rate limits.

## Keep the gateway on

`GATEWAY=1` is set in the compose file. Not for your recruiters — for crawlers.
A public URL attracts automated traffic, and the gateway stops one loop draining
the day's model budget before a person opens the link.

Issue a key for anyone who needs unrestricted runs:

```bash
curl -X POST https://<domain>/gateway/keys \
     -H 'Content-Type: application/json' \
     -d '{"tier":"pro","label":"recruiter"}'
```

## What this maps to in a company

Doing it by hand is the point — each piece has a managed equivalent you will
meet later:

| Here | At a company |
|---|---|
| Oracle Always Free VM | EC2 / Compute Engine, or EKS/GKE |
| VCN security list | Security group / firewall rule |
| Caddy auto-TLS | ALB + ACM, or cert-manager |
| `docker compose up -d` | ECS service, or a Kubernetes Deployment |
| Named volume | EBS volume, or a PersistentVolumeClaim |
| `git pull && up -d --build` | CI builds the image, GitOps deploys it |
| DuckDNS | Route 53 / Cloud DNS |
