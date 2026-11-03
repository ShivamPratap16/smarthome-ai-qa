-- name: tokens_cost_non_negative
-- engine: athena (Presto/Trino SQL), runs on Athena or offline on DuckDB over Parquet
--         (same rule as qa/data_checks/checks/09_tokens_cost_non_negative.sql)
-- severity: high
-- protects: Cost monitoring. Negative / NULL token counts or cost would break the cost dashboard.
SELECT m.id AS row_id, m.tokens_in, m.tokens_out, m.cost_usd
FROM messages m
WHERE m.tokens_in IS NULL OR m.tokens_out IS NULL OR m.cost_usd IS NULL
   OR m.tokens_in < 0 OR m.tokens_out < 0 OR m.cost_usd < 0
