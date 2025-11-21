# Test with uv
test:
	uv run pytest -s

# Run tests with latest dependencies and highest python version
test-highest:
	uv run -p 3.14 --isolated --resolution=highest pytest -s

# Run tests with oldest supported dependencies and smallest python version
test-lowest:
	uv run -p 3.10 --isolated --resolution=lowest-direct pytest -s

# Test with conda
conda-test:
	conda run --no-capture-output -n impala python -m pytest -s
