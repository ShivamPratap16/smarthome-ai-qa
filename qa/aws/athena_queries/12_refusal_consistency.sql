-- name: refusal_consistency
-- engine: athena (Presto/Trino SQL), runs on Athena or offline on DuckDB over Parquet
--         (same rule as qa/data_checks/checks/12_refusal_consistency.sql)
-- severity: high
-- protects: Safety monitoring. refused=1 must have a reason (blocked_by) or a refusal text;
--           blocked_by must be a known guardrail and always come with refused=1.
SELECT m.id AS row_id, m.refused, m.blocked_by
FROM messages m
WHERE m.role = 'assistant' AND (
      (m.refused = 1 AND m.blocked_by IS NULL
         AND lower(m.content) NOT LIKE '%can''t help%'
         AND lower(m.content) NOT LIKE '%cannot help%'
         AND lower(m.content) NOT LIKE '%can''t assist%'
         AND lower(m.content) NOT LIKE '%cannot assist%')
   OR (m.blocked_by IS NOT NULL AND m.refused <> 1)
   OR (m.blocked_by IS NOT NULL AND m.blocked_by NOT IN ('prompt_injection', 'blocked_topic', 'pii')))
