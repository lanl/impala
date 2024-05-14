# Retrieve the current package version from setup.py
version := $(shell python setup.py --version)

tag:
	git tag v$(version)
	git push --tags
