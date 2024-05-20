# Retrieve the current package version from setup.py
pkg_version := v$(shell python setup.py --version)
current_git_tag := $(shell git tag | sort -V | tail -n 1)
current_git_branch := $(shell git branch --show-current)

tag:
	if [ "$(current_git_tag)" != "$(pkg_version)" ]; then \
    if [ "$(current_git_branch)" == "master" ]; then \
      (git tag $(pkg_version) && git push --tags); \
    else \
      echo "Current branch is not 'master'. Nothing to do."; \
		fi; \
  else \
		echo "Version in 'setup.py' matches current Git tag. Nothing to do."; \
  fi

test:
	conda run --no-capture-output -n impala python -m pytest -s
