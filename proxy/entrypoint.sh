#!/bin/sh

# Create certs directory
mkdir -p /etc/nginx/certs

# Generate self-signed certificate if not exists
if [ ! -f /etc/nginx/certs/nginx.key ]; then
    echo "Generating self-signed certificate..."
    openssl req -x509 -nodes -days 3650 -newkey rsa:2048 \
        -keyout /etc/nginx/certs/nginx.key \
        -out /etc/nginx/certs/nginx.crt \
        -subj "/C=KR/ST=Seoul/L=Gangnam/O=Caloreat/CN=localhost"
fi

# Fallback to WATSON_URL if TARGET_URL is not set
if [ -z "$TARGET_URL" ]; then
    if [ -n "$WATSON_URL" ]; then
        echo "TARGET_URL not set, using WATSON_URL..."
        TARGET_URL="$WATSON_URL"
        export TARGET_URL
    else
        echo "Error: TARGET_URL or WATSON_URL environment variable is not set."
        exit 1
    fi
fi

echo "Setting up Nginx proxy to: $TARGET_URL"

# Substitute environment variables in nginx.conf
envsubst '$TARGET_URL' < /etc/nginx/nginx.conf.template > /etc/nginx/nginx.conf

# Execute passed command (nginx)
exec "$@"
