-- name: message_has_conversation
-- engine: athena (Presto/Trino SQL), runs on Athena or offline on DuckDB over Parquet
--         (same rule as qa/data_checks/checks/08_message_has_conversation.sql)
-- severity: high
-- protects: Referential integrity. No message without its conversation row.
SELECT m.id AS row_id, m.conversation_id
FROM messages m
LEFT JOIN conversations c ON c.id = m.conversation_id
WHERE c.id IS NULL
