#!/bin/bash
set -e

echo "Starting deployment script..."

# Create a combined CA bundle for Watson SDK (IAM Public Certs + Proxy Self-Signed Cert)
if [ -f "/app/proxy_certs/nginx.crt" ]; then
    echo "Creating combined CA bundle..."
    # Get location of certifi's cacert.pem
    CERTIFI_PEM=$(python -c "import certifi; print(certifi.where())")
    cat "$CERTIFI_PEM" /app/proxy_certs/nginx.crt > /app/ca_bundle.crt
    echo "Created /app/ca_bundle.crt"
else
    echo "Warning: Proxy certificate not found. Skipping CA bundle creation."
fi

# Download weights from S3
if [ -n "$MODEL_REGISTRY_BUCKET" ]; then
    echo "MODEL_REGISTRY_BUCKET is set to $MODEL_REGISTRY_BUCKET"
    python scripts/download_weights.py
else
    echo "MODEL_REGISTRY_BUCKET is not set. Using local weights if available."
fi

# Run the application
exec "$@"
