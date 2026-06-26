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
run:
	$(COMPOSE) run --rm tg2llm

status:
	docker ps | grep tg2llm
