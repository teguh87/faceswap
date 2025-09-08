# Makefile for Face Swap Advanced

.PHONY: help install install-dev install-gpu test test-cov lint format type-check clean build upload docs serve-docs

# Default target
help:
	@echo "Face Swap Advanced - Development Commands"
	@echo ""
	@echo "Installation:"
	@echo "  install      Install package in current environment"
	@echo "  install-dev  Install package with development dependencies"
	@echo "  install-gpu  Install package with GPU support"
	@echo ""
	@echo "Development:"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run linting (flake8)"
	@echo "  format       Format code (black)"
	@echo "  type-check   Run type checking (mypy)"
	@echo "  pre-commit   Run pre-commit hooks"
	@echo ""
	@echo "Build & Release:"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build package"
	@echo "  upload       Upload to PyPI"
	@echo "  upload-test  Upload to Test PyPI"
	@echo ""
	@echo "Documentation:"
	@echo "  docs         Build documentation"
	@echo "  serve-docs   Serve documentation locally"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e .[dev]

install-gpu:
	pip install -e .[gpu]

install-all:
	pip install -e .[all]

# Testing
test:
	pytest

test-cov:
	pytest --cov=face_swap_advanced --cov-report=html --cov-report=term-missing

test-verbose:
	pytest -v -s

# Code Quality
lint:
	flake8 src/ tests/

format:
	black src/ tests/

format-check:
	black --check src/ tests/

type-check:
	mypy src/

pre-commit:
	pre-commit run --all-files

# Code Quality - All
quality: format lint type-check

# Build & Release
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload: build
	python -m twine upload dist/*

upload-test: build
	python -m twine upload --repository testpypi dist/*

check-build: build
	python -m twine check dist/*

# Documentation
docs:
	cd docs && make html

serve-docs:
	cd docs/_build/html && python -m http.server 8000

# Development Environment
setup-dev: install-dev
	pre-commit install
	@echo "Development environment setup complete!"

# Virtual Environment
venv:
	python -m venv venv
	@echo "Virtual environment created. Activate with:"
	@echo "  source venv/bin/activate  (Linux/macOS)"
	@echo "  venv\\Scripts\\activate     (Windows)"

# Dependencies
update-deps:
	pip install --upgrade pip setuptools wheel
	pip install --upgrade -r requirements-dev.txt

freeze-deps:
	pip freeze > requirements-frozen.txt

# Docker
docker-build:
	docker build -t face-swap-advanced .

docker-run:
	docker run -it --rm face-swap-advanced

# Benchmarks
benchmark:
	python benchmarks/benchmark_processing.py

# Security
security:
	bandit -r src/

# Performance
profile:
	python -m cProfile -o profile.prof scripts/profile_test.py
	python -c "import pstats; pstats.Stats('profile.prof').sort_stats('cumulative').print_stats(20)"

# Examples
examples:
	python examples/basic_image_swap.py
	python examples/basic_video_swap.py

# All quality checks
ci: format-check lint type-check test-cov

# Quick development cycle
dev: format lint test

# Release preparation
release-prep: clean quality test-cov build check-build
	@echo "Release preparation complete!"
	@echo "Run 'make upload-test' to upload to Test PyPI"
	@echo "Run 'make upload' to upload to PyPI"

# Version bump (requires bump2version)
bump-patch:
	bump2version patch

bump-minor:
	bump2version minor

bump-major:
	bump2version major