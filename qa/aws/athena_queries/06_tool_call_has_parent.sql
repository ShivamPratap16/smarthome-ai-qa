-- name: tool_call_has_parent_message
-- engine: athena (Presto/Trino SQL), runs on Athena or offline on DuckDB over Parquet
--         (same rule as qa/data_checks/checks/06_tool_call_has_parent.sql)
-- severity: high
-- protects: Referential integrity. Every tool call must belong to an existing ASSISTANT message,
--           otherwise we cannot tell which answer used which device action.
SELECT tc.id AS row_id, tc.message_id, tc.name
FROM tool_calls tc
LEFT JOIN messages m ON m.id = tc.message_id
WHERE m.id IS NULL OR m.role <> 'assistant'
