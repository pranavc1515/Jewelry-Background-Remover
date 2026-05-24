# Jewelry Background Remover — Makefile
# Works on Linux/macOS and Windows (Git Bash / MSYS2).
# PowerShell users: see README.md § "Windows without make"

PYTHON   ?= python
VENV     := .venv

# Detect OS for correct venv activation path
ifeq ($(OS),Windows_NT)
    VENV_PYTHON := $(VENV)/Scripts/python
    VENV_PIP    := $(VENV)/Scripts/pip
else
    VENV_PYTHON := $(VENV)/bin/python
    VENV_PIP    := $(VENV)/bin/pip
endif

.PHONY: install ui test test-unit test-integration clean

## install  — create venv and install all dependencies
install:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	@echo ""
	@echo "Installation complete. Activate with:"
ifeq ($(OS),Windows_NT)
	@echo "  .venv\\Scripts\\Activate.ps1      (PowerShell)"
	@echo "  source .venv/Scripts/activate    (Git Bash)"
else
	@echo "  source .venv/bin/activate"
endif

## ui       — launch the Gradio web interface at http://localhost:7860
ui:
	$(VENV_PYTHON) app.py

## test     — run unit tests (no model download)
test: test-unit

test-unit:
	$(VENV_PYTHON) -m pytest tests/ -m "not integration" -v

## test-integration — run full pipeline (downloads ~170 MB u2net on first run)
test-integration:
	$(VENV_PYTHON) -m pytest tests/ -m integration -v

## clean    — remove venv and cached bytecode
clean:
	rm -rf $(VENV)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
