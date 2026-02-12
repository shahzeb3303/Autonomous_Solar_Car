# Makefile for Autonomous Solar Vehicle Project
# Usage: make <target>

.PHONY: all install install-dev install-rpi train test lint format clean deploy help

# Python interpreter
PYTHON := python3
PIP := pip3

# Directories
SRC_DIR := src
TEST_DIR := tests
TRAINING_DIR := training
MODEL_DIR := models

# Default target
all: help

# Installation targets
install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev,training]"
	pre-commit install || true

install-rpi:
	$(PIP) install -e ".[rpi]"
	# Install picamera2 via apt (not pip)
	sudo apt-get update && sudo apt-get install -y python3-picamera2

# Training targets
generate-data:
	$(PYTHON) -m training.generate_synthetic_data

train:
	$(PYTHON) -m training.train_decision_model

evaluate:
	$(PYTHON) -m training.evaluate_model

convert-tflite:
	$(PYTHON) -m training.convert_to_tflite

train-all: generate-data train evaluate convert-tflite

# Testing targets
test:
	pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing

test-fast:
	pytest $(TEST_DIR) -v -x --tb=short

test-safety:
	pytest $(TEST_DIR)/test_safety_governor.py -v

test-sensors:
	pytest $(TEST_DIR)/test_ultrasonic.py $(TEST_DIR)/test_sensor_fusion.py -v

test-integration:
	pytest $(TEST_DIR)/test_end_to_end.py -v

# Code quality targets
lint:
	flake8 $(SRC_DIR) $(TRAINING_DIR) $(TEST_DIR)
	mypy $(SRC_DIR) --ignore-missing-imports

format:
	black $(SRC_DIR) $(TRAINING_DIR) $(TEST_DIR)
	isort $(SRC_DIR) $(TRAINING_DIR) $(TEST_DIR)

format-check:
	black --check $(SRC_DIR) $(TRAINING_DIR) $(TEST_DIR)
	isort --check-only $(SRC_DIR) $(TRAINING_DIR) $(TEST_DIR)

# Benchmark and diagnostics
benchmark:
	$(PYTHON) -m scripts.benchmark_inference

diagnostics:
	$(PYTHON) -m scripts.run_diagnostics

calibrate:
	$(PYTHON) -m scripts.calibrate_sensors

# Run targets
run:
	$(PYTHON) -m src.core.main_loop

run-debug:
	LOG_LEVEL=DEBUG $(PYTHON) -m src.core.main_loop

# Deployment target (run from dev machine)
deploy:
	@echo "Deploying to Raspberry Pi..."
	@read -p "Enter RPi IP address: " RPI_IP; \
	rsync -avz --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' \
		--exclude 'notebooks' --exclude '.venv' --exclude 'venv' \
		./ pi@$$RPI_IP:~/autonomous_solar_vehicle/

# Cleanup targets
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ htmlcov/ .coverage

clean-logs:
	rm -rf logs/training/*.log logs/runtime/*.log logs/errors/*.log

clean-models:
	rm -rf $(MODEL_DIR)/saved/*.h5 $(MODEL_DIR)/tflite/*.tflite $(MODEL_DIR)/checkpoints/*

clean-data:
	rm -rf data/synthetic/*.csv data/splits/*.csv

clean-all: clean clean-logs clean-models clean-data

# Help target
help:
	@echo "Autonomous Solar Vehicle - Makefile Targets"
	@echo "============================================"
	@echo ""
	@echo "Installation:"
	@echo "  make install       - Install package (production)"
	@echo "  make install-dev   - Install with dev/training deps"
	@echo "  make install-rpi   - Install on Raspberry Pi"
	@echo ""
	@echo "Training:"
	@echo "  make generate-data - Generate synthetic training data"
	@echo "  make train         - Train the decision model"
	@echo "  make evaluate      - Evaluate trained model"
	@echo "  make convert-tflite- Convert model to TFLite"
	@echo "  make train-all     - Run full training pipeline"
	@echo ""
	@echo "Testing:"
	@echo "  make test          - Run all tests with coverage"
	@echo "  make test-fast     - Run tests (stop on first failure)"
	@echo "  make test-safety   - Run safety-critical tests"
	@echo "  make test-sensors  - Run sensor tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          - Run linters (flake8, mypy)"
	@echo "  make format        - Format code (black, isort)"
	@echo ""
	@echo "Running:"
	@echo "  make run           - Run main autonomy loop"
	@echo "  make run-debug     - Run with debug logging"
	@echo "  make diagnostics   - Run system diagnostics"
	@echo "  make calibrate     - Run sensor calibration"
	@echo "  make benchmark     - Benchmark inference speed"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy        - Deploy to Raspberry Pi via rsync"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         - Remove Python cache files"
	@echo "  make clean-all     - Remove all generated files"
