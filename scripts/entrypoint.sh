#!/bin/bash
set -e

echo "Starting deployment script..."

# Download weights from S3
if [ -n "$MODEL_REGISTRY_BUCKET" ]; then
    echo "MODEL_REGISTRY_BUCKET is set to $MODEL_REGISTRY_BUCKET"
    python scripts/download_weights.py
else
    echo "MODEL_REGISTRY_BUCKET is not set. Using local weights if available."
fi

# Run the application
exec "$@"
