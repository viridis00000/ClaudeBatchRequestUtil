#!/bin/bash

# Read Redis settings from yaml and export as environment variables
REDIS_CONFIG=$(cat /app/.devcontainer/setup_settings.yaml | yq -r '.redis')
export REDIS_PORT=$(echo "$REDIS_CONFIG" | yq -r '.port')
export REDIS_DATA_DIR=$(echo "$REDIS_CONFIG" | yq -r '.data_dir')
export REDIS_APPENDONLY=$(echo "$REDIS_CONFIG" | yq -r '.appendonly')
export REDIS_MAXMEM=$(echo "$REDIS_CONFIG" | yq -r '.maxmemory')
export REDIS_MAXMEM_POLICY=$(echo "$REDIS_CONFIG" | yq -r '.maxmemory_policy')

# Create necessary directories
mkdir -p $REDIS_DATA_DIR
mkdir -p /var/log/redis
mkdir -p /etc/supervisor/conf.d

# Copy supervisor configuration
cp /app/.devcontainer/supervisord/redis.conf /etc/supervisor/conf.d/

# Start supervisor
supervisord -c /etc/supervisor/supervisord.conf

# Wait a few seconds for Redis to start
sleep 3

# Check if Redis is running
if supervisorctl status redis | grep -q "RUNNING"; then
    echo "Redis server is running"
else
    echo "Failed to start Redis server"
    supervisorctl status redis
    exit 1
fi