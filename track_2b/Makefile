.PHONY: run run-gpu stop test

GPU = -f docker-compose.yml -f docker-compose.gpu.yml

# Starts Clausus in Docker (speech model on CPU). Works from a clean checkout once .env exists.
run: .env
	docker compose up --build

# Same, with the speech model on an NVIDIA GPU (needs the NVIDIA container toolkit).
run-gpu: .env
	docker compose $(GPU) up --build

stop:
	docker compose $(GPU) down

test:
	python -m pytest -q

.env:
	@echo "Missing .env. Run: cp .env.example .env  and set LLM_API_KEY (see README)." && exit 1
