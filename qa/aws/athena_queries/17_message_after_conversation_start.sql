-- name: message_after_conversation_start
-- engine: athena (Presto/Trino SQL), runs on Athena or offline on DuckDB over Parquet
--         (same rule as qa/data_checks/checks/17_message_after_conversation_start.sql)
-- severity: low
-- protects: Timestamp sanity. A message cannot be older than its conversation
--           (ISO-8601 UTC strings in the same format compare correctly as text).
SELECT m.id AS row_id, m.conversation_id
FROM messages m
JOIN conversations c ON c.id = m.conversation_id
WHERE m.created_at < c.started_at
