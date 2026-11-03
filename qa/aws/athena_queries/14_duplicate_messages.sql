-- name: duplicate_messages
-- engine: athena (Presto/Trino SQL), runs on Athena or offline on DuckDB over Parquet
--         (same rule as qa/data_checks/checks/14_duplicate_messages.sql)
-- severity: low
-- protects: Data quality. The same message logged twice (same conversation, role, text, second)
--           = double submit or a retry that logged twice; it inflates usage and cost numbers.
SELECT min(m.id) AS row_id, m.conversation_id, m.role, count(*) AS copies
FROM messages m
GROUP BY m.conversation_id, m.role, m.content, m.created_at
HAVING count(*) > 1
