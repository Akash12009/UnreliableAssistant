Reference observability/data stack (untested here). `docker compose -f deploy/docker-compose.yml up`, then run the API with
`UA_REDIS_URL=redis://localhost:6379/0` and the `otel_exporter` plugin endpoint `http://localhost:4317` (already the default in mcp_config.json).
Grafana: add Prometheus (http://prometheus:9090) and Tempo (http://tempo:3200) data sources; dashboards are not provided.
