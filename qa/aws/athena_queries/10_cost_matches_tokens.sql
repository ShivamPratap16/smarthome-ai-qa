-- name: cost_matches_tokens
-- engine: athena (Presto/Trino SQL), runs on Athena or offline on DuckDB over Parquet
--         (same rule as qa/data_checks/checks/10_cost_matches_tokens.sql)
-- severity: medium
-- protects: Cost monitoring. cost_usd must equal tokens_in * input price + tokens_out * output price
--           (prices are filled in from assistant/llm.py, the one place they live). Tolerance 1e-6 USD.
SELECT m.id AS row_id, m.tokens_in, m.tokens_out, m.cost_usd
FROM messages m
WHERE abs(m.cost_usd - (m.tokens_in * {{PRICE_IN_PER_TOKEN}} + m.tokens_out * {{PRICE_OUT_PER_TOKEN}})) > 0.000001
