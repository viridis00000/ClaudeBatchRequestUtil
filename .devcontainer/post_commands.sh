#!/bin/bash
echo "-----POSTCOMMAND START-----"

# Login to GitHub (can be skipped with Ctrl+C)
BROWSER=False gh auth login -h github.com -s user || true
/app/.devcontainer/set_git_conig_from_gh.sh || true


# Read Redis settings from yaml and start Redis server
REDIS_CONFIG=$(cat /app/.devcontainer/setup_settings.yaml | yq -r '.redis')
REDIS_PORT=$(echo "$REDIS_CONFIG" | yq -r '.port')
REDIS_DATA_DIR=$(echo "$REDIS_CONFIG" | yq -r '.data_dir')
REDIS_DAEMONIZE=$(echo "$REDIS_CONFIG" | yq -r '.daemonize')
REDIS_APPENDONLY=$(echo "$REDIS_CONFIG" | yq -r '.appendonly')
REDIS_MAXMEM=$(echo "$REDIS_CONFIG" | yq -r '.maxmemory')
REDIS_MAXMEM_POLICY=$(echo "$REDIS_CONFIG" | yq -r '.maxmemory_policy')

# Create Redis data directory if it doesn't exist
mkdir -p $REDIS_DATA_DIR

# Start Redis with configured settings
redis-server \
  --port $REDIS_PORT \
  --dir $REDIS_DATA_DIR \
  --daemonize $REDIS_DAEMONIZE \
  --appendonly $REDIS_APPENDONLY \
  --maxmemory $REDIS_MAXMEM \
  --maxmemory-policy $REDIS_MAXMEM_POLICY

echo "-----POSTCOMMANDS END-----"

