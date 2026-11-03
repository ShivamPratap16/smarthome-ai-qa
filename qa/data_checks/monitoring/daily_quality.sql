-- name: daily_quality
-- what: per day - bot turns, refusals, average latency, total LLM cost, conversations.
SELECT substr(m.created_at, 1, 10) AS day,
       count(*) AS turns,
       sum(CASE WHEN m.refused = 1 THEN 1 ELSE 0 END) AS refusals,
       avg(m.latency_ms) AS avg_latency_ms,
       sum(m.cost_usd) AS cost_usd,
       count(DISTINCT m.conversation_id) AS conversations
FROM messages m
WHERE m.role = 'assistant'
GROUP BY substr(m.created_at, 1, 10)
ORDER BY day
