-- ============================================================
-- QTS V8 · Smoke Test Queries
-- Quick data-integrity checks after schema init
-- ============================================================

-- [1] PostgreSQL — Schema audit
SELECT 'pg_schema_audit' AS test,
       schema_name,
       count(*) AS table_count
FROM information_schema.tables
WHERE table_type = 'BASE TABLE'
GROUP BY schema_name
ORDER BY table_count DESC;

-- [2] PostgreSQL — Key FK integrity check
SELECT 'pg_fk_orphans' AS test,
       'strategy.cross_section' AS tbl,
       count(*) AS orphan_count
FROM strategy.cross_section cs
LEFT JOIN strategy.definitions sd ON cs.strategy_id = sd.id
WHERE sd.id IS NULL;

-- [3] TimescaleDB — Hypertable verification
SELECT 'ts_hypertables' AS test,
       hypertable_schema || '.' || hypertable_name AS full_name,
       num_chunks,
       compression_enabled
FROM timescaledb_information.hypertables
ORDER BY hypertable_schema, hypertable_name;

-- [4] TimescaleDB — Data freshness check (latest tick per schema)
SELECT 'ts_freshness' AS test,
       'crypto_market' AS schema_name,
       max(exchange_time) AS latest_tick
FROM crypto_market.binance_tick
UNION ALL
SELECT 'ts_freshness', 'multi_market',
       max(exchange_time)
FROM multi_market.equity_tick
UNION ALL
SELECT 'ts_freshness', 'futures',
       max(exchange_time)
FROM futures.tick
UNION ALL
SELECT 'ts_freshness', 'prediction',
       max(recorded_at)
FROM prediction_market.tick;

-- [5] ClickHouse — Table size overview
SELECT database,
       name    AS table_name,
       total_rows,
       formatReadableSize(total_bytes) AS size_readable
FROM system.tables
WHERE database = 'qts_hist'
  AND engine != 'View'
ORDER BY total_rows DESC;

-- [6] ClickHouse — Recent ingestion count
SELECT event_date,
       count() AS parts_written
FROM system.parts
WHERE database = 'qts_hist'
  AND active = 1
  AND event_date >= today() - 7
GROUP BY event_date
ORDER BY event_date DESC;

-- [7] Kafka — Topic count (run via kafka-topics CLI)
-- kafka-topics --bootstrap-server localhost:9092 --list | wc -l

-- [8] Redis — Key pattern distribution (run via redis-cli)
-- redis-cli -a quant2024 --scan --pattern 'qts:*' | awk -F: '{print $1":"$2}' | sort | uniq -c | sort -rn | head -20
