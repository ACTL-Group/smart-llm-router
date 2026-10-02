.PHONY: up down sync

up:
	docker compose up -d
	uv sync

down:
	docker compose down

restart: down up
sync:
	uv sync