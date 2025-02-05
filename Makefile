.PHONY: install-uv run

install-uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh

run:
	uv run main.py