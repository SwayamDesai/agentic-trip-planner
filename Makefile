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

dc = cd "$(COMPOSE_DIR)" && docker compose $(ENV)

.PHONY: help up up-data down nuke logs ps health psql redis test deploy

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

up: ## app + caddy only
	$(dc) $(BASE) up -d --build
	@$(MAKE) --no-print-directory health

up-data: ## app + caddy + postgres + redis
	$(dc) $(DATA) up -d --build
	@$(MAKE) --no-print-directory health

down: ## stop, keep data
	$(dc) $(DATA) down

nuke: ## stop and delete all volumes
	$(dc) $(DATA) down -v

logs: ## follow app logs
	$(dc) $(DATA) logs -f app

ps: ## what is running
	$(dc) $(DATA) ps

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

test: ## run the test suite inside the app container
	$(dc) $(DATA) exec app python -m pytest -q

deploy: ## pull, rebuild, restart, verify
	git pull --ff-only
	$(dc) $(DATA) up -d --build
	@$(MAKE) --no-print-directory health
