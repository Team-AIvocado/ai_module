FROM --platform=linux/amd64 nginx:alpine

# Install OpenSSL and gettext (for envsubst)
RUN apk add --no-cache openssl gettext

# Script to generate certs and run nginx
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Nginx config template
COPY nginx.conf.template /etc/nginx/nginx.conf.template

ENTRYPOINT ["/entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
