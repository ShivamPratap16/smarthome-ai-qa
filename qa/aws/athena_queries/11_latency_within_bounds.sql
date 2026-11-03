-- name: latency_within_bounds
-- engine: athena (Presto/Trino SQL), runs on Athena or offline on DuckDB over Parquet
--         (same rule as qa/data_checks/checks/11_latency_within_bounds.sql)
-- severity: medium
-- protects: Latency monitoring. A bot answer must have a latency between 0 and the budget
--           ({{MAX_LATENCY_MS}} ms). Values outside mean a broken timer or a stuck request.
SELECT m.id AS row_id, m.latency_ms
FROM messages m
WHERE m.latency_ms IS NULL OR m.latency_ms < 0
   OR (m.role = 'assistant' AND m.latency_ms > {{MAX_LATENCY_MS}})
