-- name: latency_within_bounds
-- severity: medium
-- protects: Latency monitoring. A bot answer must have a latency between 0 and the budget
--           ({{MAX_LATENCY_MS}} ms). Values outside mean a broken timer or a stuck request.
SELECT m.id AS row_id, m.latency_ms
FROM messages m
WHERE m.latency_ms IS NULL OR m.latency_ms < 0
   OR (m.role = 'assistant' AND m.latency_ms > {{MAX_LATENCY_MS}})
