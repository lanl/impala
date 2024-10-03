RUN := uv run

test:
	$(RUN) python -m pytest -s

fix:
	ruff check --fix
	ruff format
