# Simple Makefile for python project set-ups

include config.mk  # Include the configuration file

# Target to set up the entire project environment
setup: install-python  create-venv configure-direnv install-packages generate-requirements init-pre-commit init-nbstripout init-nbdime

# Target to install the specified Python version (only if not installed)
install-python:
	@if ! (pyenv versions | grep -q "$(PYTHON_VERSION)"); then \
		echo "*** Installing Python $(PYTHON_VERSION)"; \
		pyenv install $(PYTHON_VERSION); \
	else \
		echo "*** Python $(PYTHON_VERSION) already installed"; \
	fi
	pyenv local $(PYTHON_VERSION)

# Target to create a virtual environment with uv
create-venv:
	@echo "*** Creating virtual environment"
	uv venv

# Target to configure direnv for automatic activation
configure-direnv:
	@echo "*** Configuring direnv"
	echo "source .venv/bin/activate" > .envrc
	direnv allow .

# Target to install packages
install-packages:
	@if [ -f requirements.txt ]; then \
		echo "*** Installing packages from requirements.txt"; \
		uv pip install -r requirements.txt; \
	else \
		echo "*** Installing packages from config.mk: $(PACKAGES)"; \
		uv pip install $(foreach pkg,$(PACKAGES),$(pkg)); \
	fi

# Target to generate requirements.txt file
generate-requirements:
	@echo "*** Generating requirements.txt"
	uv pip freeze | uv pip compile - -o requirements.txt

# initialise pre-commit hook
init-pre-commit:
	@echo "*** Installing pre-commit hooks"
	pre-commit install

# initialise nbstripout
init-nbstripout:
	@echo "*** Installing nbstripout"
	nbstripout --install

init-nbdime:
	@echo "*** Installing nbdime"
	nbdime config-git --enable
