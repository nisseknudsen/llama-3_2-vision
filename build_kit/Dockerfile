FROM ollama/ollama:0.6.5 AS builder

RUN ollama serve & \
    sleep 3 && \
    ollama pull llama3.2-vision

FROM ollama/ollama:0.6.5

COPY --from=builder /root/.ollama /root/.ollama
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY . .

RUN uv sync

COPY build_kit/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
