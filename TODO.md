# QTS V8 · Future Work & Roadmap

## Application Layer (Collectors / Streamers)

| Service | Status | Description |
|---------|--------|-------------|
| `futures_streamer` | TODO | yfinance + CFTC COT report fetcher → Kafka → TimescaleDB `futures.*` |
| `prediction_streamer` | TODO | Polymarket CLOB WebSocket + Polygon RPC → TimescaleDB `prediction_market.*` |
| `evolution_engine` | TODO | ORACLE-FORGE genetic algorithm strategy optimizer using `evolution.*` + Redis |
| `crypto_streamer` | TODO | OKX / Binance WebSocket tick stream → TimescaleDB `crypto_market.*` |
| `equity_streamer` | TODO | Multi-venue equity tick/bar → TimescaleDB `multi_market.*` |
| `feature_builder` | TODO | 5-gate + 91-dim feature pipeline → TimescaleDB `features.*` |
| `labeler` | TODO | CFL meta-labeling engine → TimescaleDB `features.crypto_fl_labels` |

### App-Layer Design Notes

**futures_streamer**
- Symbols: ES, NQ, GC, CL, ZB, NG, SI, ZC, ZS, ZW
- yfinance: real-time bars via `yfinance.Ticker.history()`
- CFTC COT: weekly PDF pull every Fri 18:00 ET
- Output: Kafka `futures.tick`, `futures.ohlcv`, `futures.cot`

**prediction_streamer**
- WebSocket: `wss://ws-subscriptions-clob.polymarket.com/ws`
- REST: `https://clob.polymarket.com`
- Max tracked markets: 100 (configurable)
- On-chain: Polygon RPC for event log parsing (resolution verification)

**evolution_engine**
- Population: 250 genomes, max 100 generations
- Tournament selection: k=7, mutation=0.15, crossover=0.60
- Elitism: 30, promote after 10 stagnant generations
- Backend: PostgreSQL `evolution.*` + Redis for live genome state

---

## Testing & CI

| Item | Priority | Description |
|------|----------|-------------|
| `tests/` directory | High | Schema validation + smoke test (created) |
| CI workflow (GitHub Actions) | Medium | `.github/workflows/validate.yml` — spin up Docker, run `init_all.sh`, execute tests |
| Data quality DAO | Medium | Per-venue data integrity checks with alerting |
| Performance benchmarks | Low | Ingest throughput, query latency benchmarks |

## Monitoring

| Item | Priority | Description |
|------|----------|-------------|
| Grafana dashboards | High | Infrastructure overview + Market health (created) |
| Alerting rules | Medium | Grafana alert rules for: data gaps > 5 min, error rate > 1%, PnL drawdown > 5% |
| Prometheus metrics | Medium | Expose app-layer metrics: tick rate, feature latency, evolution generation time |

## Security & Operations

| Item | Priority | Description |
|------|----------|-------------|
| `.env.example` | High | Template for all environment variables (created) |
| Production hardening | High | Remove hardcoded passwords, add TLS, rotate keys |
| Secret management | Medium | Integrate HashiCorp Vault or AWS Secrets Manager |
| Backup strategy | Medium | pg_dump cron + S3 lifecycle policies |
| Disaster recovery doc | Low | Recovery procedures for each storage tier |

## Schema Refinements

| Item | Priority | Description |
|------|----------|-------------|
| Partitioning strategy | Medium | Postgres `strategy.history` + `evolution.arena` time-range partitioning |
| Foreign key audit | Medium | Verify all cross-schema FK references are valid |
| Index tuning | Ongoing | Profile query patterns and add/remove indexes |
