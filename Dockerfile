FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd --system --uid 10001 recall

USER recall
# Search API + MCP over HTTP with the embedding worker. The same image also
# runs the sync daemon: `beeper-pg-sync run` (see docker-compose.yml).
ENTRYPOINT ["inbox-recall"]
CMD ["serve", "--embed", "--host", "0.0.0.0"]
