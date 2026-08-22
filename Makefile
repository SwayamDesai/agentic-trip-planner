# Layered stack. Each target brings up one more layer, and any layer can be
# omitted — the app runs on its own.
#
#   make up          app + caddy
#   make up-data     + postgres + redis
#   make down        stop everything, keep volumes
#   make nuke        stop and DELETE volumes

COMPOSE_DIR := deploy/oracle
# Quoted: the repository path may contain spaces, and an unquoted
# --env-file splits into two arguments and produces a baffling
# "unknown docker command" error.
ENV         := --env-file "$(CURDIR)/.env"
BASE        := -f docker-compose.yml
DATA        := -f docker-compose.yml -f compose.data.yml
LLM         := -f docker-compose.yml -f compose.data.yml -f compose.llm.yml
FULL        := -f docker-compose.yml -f compose.data.yml -f compose.llm.yml -f compose.langfuse.yml

dc = cd "$(COMPOSE_DIR)" && docker compose $(ENV)

.PHONY: help up up-data up-llm up-full down nuke logs ps health psql redis spend cache prompts prompts-push test deploy

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

up: ## app + caddy only
	$(dc) $(BASE) up -d --build
	@$(MAKE) --no-print-directory health

up-data: ## app + caddy + postgres + redis
	$(dc) $(DATA) up -d --build
	@$(MAKE) --no-print-directory health

up-llm: ## + litellm gateway
	$(dc) $(LLM) up -d --build
	@$(MAKE) --no-print-directory health

up-full: ## + self-hosted langfuse (heavy; cloud is lighter)
	$(dc) $(FULL) up -d --build
	@$(MAKE) --no-print-directory health
	@echo "langfuse UI: http://localhost:3000"

down: ## stop, keep data
	$(dc) $(FULL) down

nuke: ## stop and delete all volumes
	$(dc) $(FULL) down -v

logs: ## follow app logs
	$(dc) $(FULL) logs -f app

ps: ## what is running
	$(dc) $(FULL) ps

health: ## wait for the app, then report
	@for i in $$(seq 1 30); do \
	  if curl -fsS http://127.0.0.1/health >/dev/null 2>&1; then \
	    curl -s http://127.0.0.1/health; echo; exit 0; fi; \
	  sleep 2; done; \
	echo "app did not become healthy"; $(dc) $(DATA) logs --tail 30 app; exit 1

psql: ## a psql shell
	$(dc) $(DATA) exec postgres psql -U $${POSTGRES_USER:-atlas} -d $${POSTGRES_DB:-atlas}

redis: ## a redis shell
	$(dc) $(DATA) exec redis redis-cli

spend: ## what the gateway has spent, per key
	@curl -fsS -H "Authorization: Bearer $$(grep '^LITELLM_MASTER_KEY=' .env | cut -d= -f2)" \
	  http://127.0.0.1:4000/spend/logs 2>/dev/null | head -c 2000 || \
	  echo "litellm not reachable from the host (it is not published; use: make llm-exec)"

cache: ## LLM response cache: entry count and a sample
	@$(dc) $(LLM) exec redis sh -c '\
	  echo "entries: $$(redis-cli --scan --pattern "litellm.cache*" | wc -l)"; \
	  echo "keys without a TTL (rate-limit counters, must not be evicted):"; \
	  redis-cli --scan --pattern "*" | head -20 | while read k; do \
	    t=$$(redis-cli ttl "$$k"); [ "$$t" = "-1" ] && echo "  $$k"; done; true'

prompts: ## what each prompt resolves to, and from where
	$(dc) $(FULL) exec app python -m providers.prompts status

prompts-push: ## seed the Langfuse prompt registry from the shipped defaults
	$(dc) $(FULL) exec app python -m providers.prompts

test: ## run the test suite inside the app container
	$(dc) $(DATA) exec app python -m pytest -q

deploy: ## pull, rebuild, restart, verify
	git pull --ff-only
	$(dc) $(DATA) up -d --build
	@$(MAKE) --no-print-directory health
