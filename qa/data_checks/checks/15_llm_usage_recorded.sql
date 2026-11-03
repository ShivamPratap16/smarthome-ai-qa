-- name: llm_usage_recorded
-- severity: medium
-- protects: Cost monitoring. A bot answer that was NOT blocked by a guardrail went through the LLM,
--           so it must have tokens_in > 0 (otherwise cost is silently under-reported).
SELECT m.id AS row_id, m.tokens_in
FROM messages m
WHERE m.role = 'assistant' AND m.blocked_by IS NULL AND m.tokens_in = 0
