COMPOSE = docker compose -f docker-compose.yml

.PHONY: up down logs build run status

# Start the GPU vision server (llama.cpp + Qwen2.5-VL on Vulkan).
up:
	$(COMPOSE) up -d vision

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f vision

# Build the tg2llm utility image (Go parse binary + whisper-cli + ffmpeg).
build:
	$(COMPOSE) build tg2llm

# Run the parser against the export (vision via GPU server, voice via bundled whisper-cli).
# Override paths with env vars, e.g.: make run EXPORT_DIR=/path/to/ChatExport
# TG2LLM_UID/GID keep the shared ../cache and the output file owned by you, not root.
run:
	mkdir -p -m 700 ../cache
	TG2LLM_UID=$$(id -u) TG2LLM_GID=$$(id -g) $(COMPOSE) run --rm tg2llm

status:
	docker ps | grep tg2llm
