# Load settings from YAML
SETTINGS_FILE := src/python/anthropic_batch_request_util/config/settings.yaml
PYTHON_LOAD_SETTINGS := python -c 'import yaml; print(yaml.safe_load(open("$(SETTINGS_FILE)"))["worker"]["task"]["default_queue"])'
WORKER_QUEUE := $(shell $(PYTHON_LOAD_SETTINGS))

# Environment variables
PYTHON_PATH := PYTHONPATH=.
CELERY_APP := src.python.anthropic_batch_request_util.batch_tasks:celery_app
WORKER_CONCURRENCY := 1

# Directories from settings
LOG_DIR := logs
OUTPUT_DIR := output

.PHONY: start-worker stop-worker start-worker-debug test clean help

# Celery worker related commands
start-worker:
	@echo "Starting Celery worker..."
	@$(PYTHON_PATH) celery -A $(CELERY_APP) worker \
		-Q $(WORKER_QUEUE) \
		-c $(WORKER_CONCURRENCY) \
		--loglevel=INFO \
		-n anthropic_worker@%h \
		--logfile=${OUTPUT_DIR}/$(LOG_DIR)/celery.log \
		-D

start-worker-debug:
	@echo "Starting Celery worker in debug mode..."
	@$(PYTHON_PATH) celery -A $(CELERY_APP) worker \
		-Q $(WORKER_QUEUE) \
		-c $(WORKER_CONCURRENCY) \
		--loglevel=DEBUG \
		-n anthropic_worker@%h

stop-worker:
	@echo "Stopping Celery worker..."
	@pkill -f "celery.*anthropic_worker" || true

# Test related commands
test:
	@echo "Running test request..."
	@$(PYTHON_PATH) python -m src.python.anthropic_batch_request_util.tools.test_batch_request

run-batch: check-json
	@echo "Running batch request from: $(json_file)"
	@$(PYTHON_PATH) python -c "import json; \
		from src.python.anthropic_batch_request_util.batch_handler import AnthropicBatchHandler; \
		with open('$(json_file)', 'r') as f: \
			data = json.load(f); \
		handler = AnthropicBatchHandler(); \
		result = handler.execute_batch_with_monitoring( \
			system_prompt=data['system_prompt'], \
			messages_list=data['messages_list'], \
			custom_id_prefix=data.get('custom_id_prefix', 'request') \
		); \
		print(json.dumps(result, indent=2))"

check-json:
	@if [ -z "$(json_file)" ]; then \
		echo "Error: json_file parameter is required"; \
		echo "Usage: make run-batch json_file=path/to/request.json"; \
		exit 1; \
	fi

# Cleanup command
clean:
	@echo "Cleaning temporary files and logs..."
	@rm -rf $(LOG_DIR)/* $(OUTPUT_DIR)/*
	@echo "Cleanup complete"

# Help command
help:
	@echo "Available commands:"
	@echo ""
	@echo "Worker management:"
	@echo "  make start-worker       - Start Celery worker in background"
	@echo "  make start-worker-debug - Start Celery worker in debug mode"
	@echo "  make stop-worker        - Stop Celery worker"
	@echo ""
	@echo "Testing and execution:"
	@echo "  make test               - Run test request"
	@echo "  make run-batch json_file=path/to/request.json - Run batch request from JSON file"
	@echo ""
	@echo "Other:"
	@echo "  make clean              - Clean temporary files and logs"
	@echo "  make help               - Show this help message"

# Default target
.DEFAULT_GOAL := help 