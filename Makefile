# Test with uv
test:
	uv run pytest -s

# Test with conda
conda-test:
	conda run --no-capture-output -n impala python -m pytest -s
