#!/bin/bash
set -e

echo "[System] 배포 스크립트를 시작합니다..."

# Create a combined CA bundle for Watson SDK (IAM Public Certs + Proxy Self-Signed Cert)
# Check if we need to wait for proxy certificate
if [[ "$WATSON_URL" == *"127.0.0.1"* ]] || [[ "$WATSON_URL" == *"ai-proxy"* ]]; then
    echo "[System] 프록시 모드 감지. 인증서를 대기합니다: /app/proxy_certs/nginx.crt"
    
    # Wait up to 30 seconds for the certificate
    TIMEOUT=30
    COUNTER=0
    while [ ! -f "/app/proxy_certs/nginx.crt" ]; do
        if [ $COUNTER -ge $TIMEOUT ]; then
            echo "[Error] 30초 내에 인증서를 찾지 못했습니다."
            break
        fi
        echo "[System] 인증서 대기 중... ($COUNTER/$TIMEOUT)"
        sleep 1
        COUNTER=$((COUNTER+1))
    done

    if [ -f "/app/proxy_certs/nginx.crt" ]; then
        echo "[System] 인증서 발견. CA 번들을 생성합니다..."
        CERTIFI_PEM=$(python -c "import certifi; print(certifi.where())")
        cat "$CERTIFI_PEM" /app/proxy_certs/nginx.crt > /tmp/ca_bundle.crt
        echo "[System] CA 번들 생성 완료: /tmp/ca_bundle.crt"
        # Export the new bundle path for requests and boto3
        export REQUESTS_CA_BUNDLE=/tmp/ca_bundle.crt
        export AWS_CA_BUNDLE=/tmp/ca_bundle.crt
        ls -l /tmp/ca_bundle.crt
    else
        echo "[Warning] CA 번들 생성 실패. 인증 문제가 발생할 수 있습니다."
    fi
else
    echo "[System] 프록시 모드를 사용하지 않습니다 (URL=$WATSON_URL). CA 번들 생성을 건너뜁니다."
fi

# S3 가중치 다운로드 (일괄 다운로드 중단: model_loader.py에서 동적 다운로드 처리)
# if [ -n "$MODEL_REGISTRY_BUCKET" ]; then
#     echo "MODEL_REGISTRY_BUCKET 환경 변수가 설정되었습니다: $MODEL_REGISTRY_BUCKET"
#     # python scripts/download_weights.py
# else
#     echo "MODEL_REGISTRY_BUCKET이 설정되지 않았습니다. 로컬 가중치를 사용합니다."
# fi
echo "[System] 추론 모듈 시작 준비 완료."

# Run the application
exec "$@"
