-- name: llm_usage_recorded
-- engine: athena (Presto/Trino SQL), runs on Athena or offline on DuckDB over Parquet
--         (same rule as qa/data_checks/checks/15_llm_usage_recorded.sql)
-- severity: medium
-- protects: Cost monitoring. A bot answer that was NOT blocked by a guardrail went through the LLM,
--           so it must have tokens_in > 0 (otherwise cost is silently under-reported).
SELECT m.id AS row_id, m.tokens_in
FROM messages m
WHERE m.role = 'assistant' AND m.blocked_by IS NULL AND m.tokens_in = 0
