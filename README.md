# Anthropic Batch Request Utility

[日本語](README.ja.md) | [English](README.md)

## Overview
A utility for making batch requests to the Anthropic API using Celery workers for distributed processing and monitoring. This tool enables efficient handling of large-scale requests with features like automatic retries, progress monitoring, and result management.

## Requirements
- Python 3.12+
- Docker and Docker Compose (strongly recommended)
- Redis server (optional, defaults to Redis in Docker)
- Anthropic API key
- VS Code with Dev Containers extension (for development)

## Installation

### Using Dev Container (Recommended)
1. Clone the repository
2. Open in VS Code with Dev Containers extension installed
3. Click "Reopen in Container" when prompted
4. Wait for container build and initialization
5. Create `.env` file from template and set your API key:
   ```bash
   cp .env.back .env
   # Edit .env and set your Anthropic API key:
   # ANTHROPIC_API_KEY=your_api_key_here
   ```

### Manual Installation
1. Install Python 3.12+
2. Install Redis server
3. Clone the repository
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Create and configure `.env` file as described above

## Configuration
Configure the tool editing `src/python/anthropic_batch_request_util/config/settings.yaml`:

### API Configuration
```yaml
api:
  version: "2023-06-01"                # Anthropic API version
  beta_features:                       # Beta features to enable
    - "message-batches-2024-09-24"
    - "prompt-caching-2024-07-31"
  timeout: 300                         # Request timeout in seconds
  max_payload_size: 104857600          # Maximum payload size (100MB)
```

### Model Configuration
```yaml
model:
  name: "claude-3-5-haiku-20241022"    # Model identifier
  display_name: "haiku"                # Display name for logging
  max_tokens: 8192                     # Maximum output tokens
  temperature: 0.0                     # Generation temperature (0-1)
  response_format: "text"              # Response format (text/json)
```

### Batch Processing
```yaml
batch:
  max_size: 10000                      # Maximum requests per batch
  chunk_size: 1000                     # Requests per chunk for large batches
  enable_prompt_cache: false           # Enable prompt caching
  cache_type: "ephemeral"              # Cache type (ephemeral only)
```

### Storage Configuration
```yaml
storage:
  base_dir: "output"                   # Base directory for outputs
  subdirs:
    requests: "batch_records"          # Request records location
    results: "results"                 # Results location
    logs: "logs"                       # Log files location
```

## Usage

### Worker Management
Manage Celery workers using the following make commands:

```bash
# Start worker in background mode
make start-worker

# Start worker in debug mode with console output
make start-worker-debug

# Stop running worker
make stop-worker
```

### Batch Processing

1. Create a batch request JSON file (see examples/batch_request.json):
```json
{
  "system_prompt": "You are a helpful AI assistant. Please provide clear and concise answers.",
  "messages_list": [
    [{"role": "user", "content": "What is 2+2?"}],
    [{"role": "user", "content": "What is the capital of France?"}]
  ],
  "custom_id_prefix": "example_batch"
}
```

2. Run batch request:
```bash
make run-batch json_file=path/to/request.json
```

### Testing
Run the test suite with sample requests:
```bash
make test
```

### Other Commands
```bash
# Clean temporary files and logs
make clean

# Show all available commands
make help
```

Results and logs will be stored in:
- Batch records: `output/batch_records/`
- Results: `output/results/`
- Worker logs: `logs/celery.log`

