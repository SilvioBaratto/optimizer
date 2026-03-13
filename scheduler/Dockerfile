# Lightweight scheduler — Alpine + curl + supercronic (~12 MB)
# All fetch logic uses curl to hit the API; no Python needed.
FROM alpine:3.21

ARG TARGETARCH=amd64
ARG SUPERCRONIC_VERSION=v0.2.33

# Install curl (for HTTP calls) and tzdata (for timezone-aware logging)
RUN apk add --no-cache curl jq tzdata \
    && ARCH=$(case "${TARGETARCH}" in arm64) echo "arm64" ;; *) echo "amd64" ;; esac) \
    && curl -fsSL "https://github.com/aptible/supercronic/releases/download/${SUPERCRONIC_VERSION}/supercronic-linux-${ARCH}" \
       -o /usr/local/bin/supercronic \
    && chmod +x /usr/local/bin/supercronic

# Non-root user
RUN adduser -D -u 1000 scheduler

# Copy scheduler assets
COPY scheduler/crontab /etc/supercronic/crontab
COPY scheduler/fetch.sh /app/fetch.sh
RUN chmod +x /app/fetch.sh && chown -R scheduler:scheduler /app

USER scheduler

CMD ["supercronic", "/etc/supercronic/crontab"]
