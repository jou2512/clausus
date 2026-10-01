.PHONY: run stop test

# Starts Clausus in Docker. Works from a clean checkout once .env exists.
run: .env
	docker compose up --build

stop:
	docker compose down

test:
	python -m pytest -q

.env:
	@echo "Missing .env. Run: cp .env.example .env  and set LLM_API_KEY (see README)." && exit 1
