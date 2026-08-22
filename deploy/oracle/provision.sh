#!/usr/bin/env bash
# Provision a fresh Oracle Always Free VM (Ubuntu 22.04/24.04, arm64) to run
# Atlas. Idempotent — safe to re-run.
#
#   curl -fsSL <raw url>/provision.sh | bash
# or
#   ./provision.sh
set -euo pipefail

REPO="${REPO:-https://github.com/SwayamDesai/agentic-trip-planner.git}"
DIR="${DIR:-$HOME/atlas}"

say() { printf '\n\033[1;34m==>\033[0m %s\n' "$1"; }

say "Installing Docker"
if ! command -v docker >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq ca-certificates curl git
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin
  sudo usermod -aG docker "$USER"
else
  echo "already installed"
fi

# Docker starts on boot, so the app survives a VM reboot without a systemd unit
sudo systemctl enable --now docker

say "Opening ports 80 and 443 on the instance firewall"
# The Oracle gotcha: Ubuntu images ship iptables rules that drop everything
# except SSH, and opening the VCN security list in the console is NOT enough.
# Both layers must allow the traffic. This is the single most common reason an
# Oracle VM appears unreachable.
sudo iptables -C INPUT -p tcp --dport 80 -j ACCEPT 2>/dev/null \
  || sudo iptables -I INPUT 6 -p tcp --dport 80 -j ACCEPT
sudo iptables -C INPUT -p tcp --dport 443 -j ACCEPT 2>/dev/null \
  || sudo iptables -I INPUT 6 -p tcp --dport 443 -j ACCEPT
sudo apt-get install -y -qq iptables-persistent >/dev/null 2>&1 || true
sudo netfilter-persistent save >/dev/null 2>&1 || true

say "Adding swap"
# The Always Free ARM shape has plenty of RAM, but a Docker build of the
# dependency tree is memory-spiky and an OOM mid-build is a confusing failure.
if ! sudo swapon --show | grep -q /swapfile; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
else
  echo "already present"
fi

say "Fetching the code"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" pull --ff-only
else
  git clone --depth 1 "$REPO" "$DIR"
fi

say "Next steps"
cat <<'NEXT'
1. Write your secrets:

     nano ~/atlas/.env

   Required:
     GROQ_API_KEY=...
     GROQ_API_KEY_2=...
     GROQ_API_KEY_3=...
     SERPAPI_KEY=...
     GATEWAY_KEY_SALT=<any stable random string>

   Optional, and what gives you HTTPS:
     DOMAIN=atlas-yourname.duckdns.org

   With no DOMAIN the site serves plain HTTP on port 80.

2. Start it:

     cd ~/atlas/deploy/oracle
     docker compose up -d --build

   The first build takes a few minutes on ARM.

3. Check it:

     curl localhost/health
     docker compose logs -f

If you added DOMAIN, point that DNS record at this VM's public IP first —
Caddy needs to answer a challenge on it before it can issue a certificate.

You may need to log out and back in for docker group membership to apply.
NEXT
