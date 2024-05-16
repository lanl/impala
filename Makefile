# Retrieve the current package version from setup.py
pkg_version := v$(shell python setup.py --version)
current_git_tag := $(shell git tag | sort -V | tail -n 1)

tag:
	if [ $(current_git_tag) != $(pkg_version) ]; then (git tag $(pkg_version) && git push --tags); fi

test:
	conda run --no-capture-output -n impala python -m pytest -s
