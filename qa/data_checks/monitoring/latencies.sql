-- name: latencies
-- what: latency of every bot answer that reached the LLM (p95 is computed in Python,
--       because SQLite has no percentile function).
SELECT substr(m.created_at, 1, 10) AS day, m.latency_ms
FROM messages m
WHERE m.role = 'assistant' AND m.blocked_by IS NULL
ORDER BY m.id
