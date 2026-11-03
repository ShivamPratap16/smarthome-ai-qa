-- name: tokens_cost_non_negative
-- severity: high
-- protects: Cost monitoring. Negative / NULL token counts or cost would break the cost dashboard.
SELECT m.id AS row_id, m.tokens_in, m.tokens_out, m.cost_usd
FROM messages m
WHERE m.tokens_in IS NULL OR m.tokens_out IS NULL OR m.cost_usd IS NULL
   OR m.tokens_in < 0 OR m.tokens_out < 0 OR m.cost_usd < 0
