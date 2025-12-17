#!/bin/bash
set -e

echo "Starting deployment script..."

# Create a combined CA bundle for Watson SDK (IAM Public Certs + Proxy Self-Signed Cert)
# Check if we need to wait for proxy certificate
if [[ "$WATSON_URL" == *"127.0.0.1"* ]] || [[ "$WATSON_URL" == *"ai-proxy"* ]]; then
    echo "Using Proxy mode. Waiting for certificate at /app/proxy_certs/nginx.crt..."
    
    # Wait up to 30 seconds for the certificate
    TIMEOUT=30
    COUNTER=0
    while [ ! -f "/app/proxy_certs/nginx.crt" ]; do
        if [ $COUNTER -ge $TIMEOUT ]; then
            echo "Error: Certificate not found after $TIMEOUT seconds."
            break
        fi
        echo "Waiting for certificate... ($COUNTER/$TIMEOUT)"
        sleep 1
        COUNTER=$((COUNTER+1))
    done

    if [ -f "/app/proxy_certs/nginx.crt" ]; then
        echo "Certificate found. Creating combined CA bundle..."
        CERTIFI_PEM=$(python -c "import certifi; print(certifi.where())")
        cat "$CERTIFI_PEM" /app/proxy_certs/nginx.crt > /tmp/ca_bundle.crt
        echo "Created /tmp/ca_bundle.crt"
        ls -l /tmp/ca_bundle.crt
    else
        echo "Warning: Failed to create CA bundle. Authentication may fail."
    fi
else
    echo "Not using Proxy mode (WATSON_URL=$WATSON_URL). Skipping CA bundle creation."
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
