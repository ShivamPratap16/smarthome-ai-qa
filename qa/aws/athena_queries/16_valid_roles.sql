-- name: valid_roles
-- engine: athena (Presto/Trino SQL), runs on Athena or offline on DuckDB over Parquet
--         (same rule as qa/data_checks/checks/16_valid_roles.sql)
-- severity: medium
-- protects: Schema contract. Only 'user' and 'assistant' messages are logged.
SELECT m.id AS row_id, m.role
FROM messages m
WHERE m.role IS NULL OR m.role NOT IN ('user', 'assistant')
