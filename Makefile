# ─────────────────────────────────────────────────────────────────
# llm-bawt Makefile — cross-platform (Linux / macOS / Windows)
# ─────────────────────────────────────────────────────────────────
#
# Workflow:
#   make install         Editable uv tool install — CLI client from local repo
#   make up / docker-dev Start the service stack in Docker
#   make dev             Sync local .venv for source development
#   make test / lint     Run tests and linting (uv-based)
#
# ─────────────────────────────────────────────────────────────────

.DEFAULT_GOAL := help

# ── OS detection ────────────────────────────────────────────────
ifeq ($(OS),Windows_NT)
    DETECTED_OS  := Windows
    SHELL        := cmd.exe
    PYTHON       := python
    RM_RF        := rmdir /s /q
    MKDIR        := mkdir
    NULL         := NUL
    SEP          := \\
    # Windows: no color by default
    BLUE         :=
    GREEN        :=
    YELLOW       :=
    RED          :=
    NC           :=
    AND          := &
    CHECK_CMD     = where $(1) >$(NULL) 2>&1
    FIND_PID      = $(error PID-based service management not supported on Windows — use Docker)
else
    DETECTED_OS  := $(shell uname -s)
    PYTHON       := python3
    RM_RF        := rm -rf
    MKDIR         = mkdir -p
    NULL         := /dev/null
    SEP          := /
    BLUE         := \033[0;34m
    GREEN        := \033[0;32m
    YELLOW       := \033[1;33m
    RED          := \033[0;31m
    NC           := \033[0m
    AND          := &&
    CHECK_CMD     = command -v $(1) >$(NULL) 2>&1
    FIND_PID      = cat $(1) 2>$(NULL)
endif

# ── Configurable variables ──────────────────────────────────────
REPO           ?= git+https://github.com/bawthub/llm-bawt.git
CUDA_ARCHS     ?= 75;80;86;89;90;120  # used by dev-llama
WITH_CUDA      ?= true                # used by dev-llama

# Docker
COMPOSE_BASE   := docker-compose.yml
COMPOSE_DEV    := docker-compose.dev.yml

# Local servers
MCP_HOST    ?= 0.0.0.0
MCP_PORT    ?= 8001
SERVICE_HOST   ?= 0.0.0.0
SERVICE_PORT   ?= 8642
MCP_URL     := http://127.0.0.1:$(MCP_PORT)
RUN_DIR        := .run
LOG_DIR        := .logs
MCP_PID     := $(RUN_DIR)/mcp-server.pid
SERVICE_PID    := $(RUN_DIR)/llm-service.pid

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

.PHONY: help
help: ## Show this help
	@echo ""
	@echo "llm-bawt — available targets"
	@echo "════════════════════════════════════════════════════"
ifeq ($(DETECTED_OS),Windows)
	@findstr /R /C:"^[a-zA-Z_-]*:.*##" $(MAKEFILE_LIST)
else
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*##"}; {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
endif

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INSTALLATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

.PHONY: check-python check-uv

check-python: ## Verify Python 3.12+
	@$(PYTHON) -c "import sys; exit(0 if sys.version_info >= (3,12) else 1)" \
		|| (echo "$(RED)Python 3.12+ required$(NC)" && exit 1)
	@echo "$(GREEN)✓ Python OK$(NC)"

check-uv: ## Verify uv is installed
	@$(call CHECK_CMD,uv) || (echo "$(RED)uv not found — install: https://docs.astral.sh/uv/$(NC)" && exit 1)
	@echo "$(GREEN)✓ uv OK$(NC)"


.PHONY: install install-github uninstall

install: check-python check-uv ## [uv tool] Editable install from local repo (recommended)
	@echo "$(BLUE)Installing llm-bawt via uv tool (editable, local repo)...$(NC)"
	-uv tool uninstall llm-bawt 2>$(NULL) || true
	uv tool install --force --editable .
	@echo "$(GREEN)✓ llm-bawt installed (editable)$(NC)"
	@echo "  Run: llm --status"
	@echo "  Source changes in this repo are reflected immediately"
	@echo "  Service deps (memory, search, etc.) run in Docker — see: make up"

install-github: check-python check-uv ## [uv tool] Non-editable install from GitHub
	@echo "$(BLUE)Installing llm-bawt via uv tool from GitHub...$(NC)"
	-uv tool uninstall llm-bawt 2>$(NULL) || true
	uv tool install --force "$(REPO)"
	@echo "$(GREEN)✓ llm-bawt installed (GitHub)$(NC)"

# To add extra packages into the tool env (rarely needed):
#   uv tool install --with <package> llm-bawt

uninstall: ## [uv tool] Uninstall llm-bawt
	@echo "$(YELLOW)Uninstalling llm-bawt...$(NC)"
	-uv tool uninstall llm-bawt 2>$(NULL) || true
	@echo "$(GREEN)✓ llm-bawt uninstalled$(NC)"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEVELOPMENT (uv-based local .venv)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

.PHONY: dev dev-llama dev-run

dev: check-uv ## [uv] Sync .venv with all extras (dev, mcp, service, search, memory, hf, tui)
	@echo "$(BLUE)Syncing .venv with all extras...$(NC)"
	uv sync --inexact --extra dev --extra mcp --extra service --extra search --extra memory --extra huggingface
	@echo "$(GREEN)✓ .venv synced$(NC)"
	@echo "  Run: uv run llm --status"

dev-llama: check-uv ## [uv] Add llama-cpp-python to .venv (CUDA-aware)
ifeq ($(WITH_CUDA),true)
	@echo "$(BLUE)Installing llama-cpp-python with CUDA into .venv...$(NC)"
	CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCHS)" \
		uv pip install llama-cpp-python --reinstall
else
	@echo "$(BLUE)Installing llama-cpp-python (CPU) into .venv...$(NC)"
	uv pip install llama-cpp-python
endif
	@echo "$(GREEN)✓ llama-cpp-python installed$(NC)"

dev-vllm: check-uv ## [uv] Add vLLM to .venv (requires NVIDIA GPU + CUDA)
	@echo "$(BLUE)Installing vLLM into .venv...$(NC)"
	uv pip install vllm
	@echo "$(GREEN)✓ vLLM installed$(NC)"
	@echo "$(YELLOW)Note: vLLM requires NVIDIA GPU + CUDA to run$(NC)"

dev-run: ## [uv] Run llm from .venv (e.g. make dev-run ARGS="--status")
	uv run llm $(ARGS)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DOCKER (works identically on Linux / macOS / Windows)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

.PHONY: run up docker-dev down restart restart-app rebuild logs docker-status docker-shell docker-exec

run: ## Stop app + bridges, rebuild dev, tail logs
	docker compose -f $(COMPOSE_BASE) -f $(COMPOSE_DEV) down --remove-orphans --timeout 5 app openclaw-bridge claude-code-bridge
	docker compose -f $(COMPOSE_BASE) -f $(COMPOSE_DEV) up -d app openclaw-bridge claude-code-bridge
	docker compose logs -f --tail=50 app openclaw-bridge claude-code-bridge

up: ## Docker compose up (production)
	@echo "Starting containers (production mode)..."
	docker compose up -d
	@echo "$(GREEN)✓ Services started on ports $(MCP_PORT) (MCP) and $(SERVICE_PORT) (LLM)$(NC)"

docker-dev: ## Docker compose up (dev mode — mounts ./src)
	@echo "Starting containers (dev mode)..."
	docker compose -f $(COMPOSE_BASE) -f $(COMPOSE_DEV) up -d
	@echo "$(GREEN)✓ Dev mode started — src/ mounted live$(NC)"

down: ## Docker compose down
	docker compose down
	@echo "$(GREEN)✓ Containers stopped$(NC)"

restart: ## Restart containers
	docker compose restart
	@echo "$(GREEN)✓ Containers restarted$(NC)"

restart-app: ## Restart ONLY the app (leaves bridges/agents alive)
	docker compose -f $(COMPOSE_BASE) -f $(COMPOSE_DEV) restart app
	@echo "$(GREEN)✓ app restarted (bridges untouched)$(NC)"

rebuild: ## Rebuild and restart containers
	docker compose down
	docker compose up -d --build
	@echo "$(GREEN)✓ Containers rebuilt and started$(NC)"

logs: ## Follow container logs
	docker compose logs -f --tail=50

docker-status: ## Show container status
	docker compose ps

docker-shell: ## Open bash shell in app container
	docker compose exec app bash

docker-exec: ## Run command in container (make docker-exec CMD="llm --status")
	docker compose exec app $(CMD)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOCAL SERVERS (uv-based — Linux / macOS only)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

.PHONY: server-start server-stop server-restart server-status

ifeq ($(DETECTED_OS),Windows)

server-start server-stop server-restart server-status:
	@echo "$(RED)Local server management requires Linux/macOS. Use Docker targets instead (make up / make down).$(NC)"

else

$(RUN_DIR) $(LOG_DIR):
	@$(MKDIR) $@

server-start: check-uv | $(RUN_DIR) $(LOG_DIR) ## Start MCP + LLM services (background)
	@# --- Memory/MCP server ---
	@if [ -f "$(MCP_PID)" ] && kill -0 $$(cat "$(MCP_PID)") 2>$(NULL); then \
		echo "MCP server already running (pid $$(cat $(MCP_PID)))"; \
	else \
		echo "$(BLUE)[mcp] Starting on $(MCP_HOST):$(MCP_PORT)...$(NC)"; \
		LLM_BAWT_MCP_SERVER_VERBOSE= \
		nohup uv run --extra mcp llm-mcp-server --transport http \
			--host $(MCP_HOST) --port $(MCP_PORT) \
			> $(LOG_DIR)/mcp-server.log 2>&1 & \
		echo $$! > $(MCP_PID); \
	fi
	@# --- Wait for MCP to be ready ---
	@echo "$(BLUE)[llm] Waiting for MCP server...$(NC)"
	@for i in $$(seq 1 20); do \
		if $(PYTHON) -c "import socket; s=socket.socket(); s.settimeout(0.5); exit(0 if s.connect_ex(('127.0.0.1',$(MCP_PORT)))==0 else 1)" 2>$(NULL); then \
			echo "$(GREEN)[llm] MCP server ready$(NC)"; \
			break; \
		fi; \
		sleep 0.5; \
	done
	@# --- LLM service ---
	@if [ -f "$(SERVICE_PID)" ] && kill -0 $$(cat "$(SERVICE_PID)") 2>$(NULL); then \
		echo "llm-service already running (pid $$(cat $(SERVICE_PID)))"; \
	else \
		echo "$(BLUE)[llm] Starting on $(SERVICE_HOST):$(SERVICE_PORT)...$(NC)"; \
		LLM_BAWT_MCP_SERVER_URL="$(MCP_URL)" \
		nohup uv run --extra service --extra search --extra memory llm-service \
			--host $(SERVICE_HOST) --port $(SERVICE_PORT) \
			> $(LOG_DIR)/llm-service.log 2>&1 & \
		echo $$! > $(SERVICE_PID); \
		echo "$(GREEN)✓ Services started$(NC)"; \
	fi
	@$(MAKE) --no-print-directory server-status

server-stop: ## Stop MCP + LLM services
	@if [ -f "$(SERVICE_PID)" ] && kill -0 $$(cat "$(SERVICE_PID)") 2>$(NULL); then \
		echo "Stopping llm-service (pid $$(cat $(SERVICE_PID)))..."; \
		kill $$(cat "$(SERVICE_PID)") 2>$(NULL) || true; \
		sleep 1; \
		kill -0 $$(cat "$(SERVICE_PID)") 2>$(NULL) && kill -9 $$(cat "$(SERVICE_PID)") 2>$(NULL) || true; \
	else \
		echo "llm-service not running"; \
	fi
	@rm -f "$(SERVICE_PID)"
	@if [ -f "$(MCP_PID)" ] && kill -0 $$(cat "$(MCP_PID)") 2>$(NULL); then \
		echo "Stopping MCP server (pid $$(cat $(MCP_PID)))..."; \
		kill $$(cat "$(MCP_PID)") 2>$(NULL) || true; \
		sleep 1; \
		kill -0 $$(cat "$(MCP_PID)") 2>$(NULL) && kill -9 $$(cat "$(MCP_PID)") 2>$(NULL) || true; \
	else \
		echo "MCP server not running"; \
	fi
	@rm -f "$(MCP_PID)"
	@echo "$(GREEN)✓ Services stopped$(NC)"

server-restart: server-stop server-start ## Restart local services

server-status: ## Show local service status
	@if [ -f "$(MCP_PID)" ] && kill -0 $$(cat "$(MCP_PID)") 2>$(NULL); then \
		echo "MCP server: $(GREEN)running$(NC) (pid $$(cat $(MCP_PID))) on port $(MCP_PORT)"; \
	else \
		echo "MCP server: $(RED)stopped$(NC)"; \
	fi
	@if [ -f "$(SERVICE_PID)" ] && kill -0 $$(cat "$(SERVICE_PID)") 2>$(NULL); then \
		echo "llm-service:   $(GREEN)running$(NC) (pid $$(cat $(SERVICE_PID))) on port $(SERVICE_PORT)"; \
	else \
		echo "llm-service:   $(RED)stopped$(NC)"; \
	fi
	@echo "Logs: $(LOG_DIR)/"

endif  # not Windows

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEV TOOLS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

.PHONY: lint format typecheck test test-cov test-integration test-all

lint: ## Lint with ruff
	uv run ruff check src/

format: ## Format with ruff
	uv run ruff format src/

typecheck: ## Type-check with mypy
	uv run mypy src/

test: ## Run hermetic tests only (fast, no DB/Redis/live service needed)
	uv run pytest

test-cov: ## Run hermetic tests with coverage
	uv run pytest --cov=src --cov-report=term-missing

test-integration: ## Run tests marked `integration` (need live Postgres/Redis) — pair with plain `test`
	uv run pytest -m integration

test-all: ## Run EVERY test (hermetic + integration + service + llm_call). Use before merging inter-bot / dispatcher / capability / bridge changes — a bare `pytest` silently skips those (TASK-780).
	uv run pytest -m ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLEAN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

.PHONY: clean clean-all

clean: ## Remove build artifacts and caches
	$(RM_RF) dist build *.egg-info .pytest_cache .mypy_cache .ruff_cache
ifeq ($(DETECTED_OS),Windows)
	-for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
else
	find . -type d -name __pycache__ -exec rm -rf {} + 2>$(NULL) || true
endif
	@echo "$(GREEN)✓ Cleaned$(NC)"

clean-all: clean ## Remove .venv, logs, and run dirs too
	$(RM_RF) .venv $(LOG_DIR) $(RUN_DIR)
	@echo "$(GREEN)✓ Deep cleaned$(NC)"
