RUN := uv run

test:
	$(RUN) python -m pytest -s

fmt:
	ruff check
	ruff format --diff
