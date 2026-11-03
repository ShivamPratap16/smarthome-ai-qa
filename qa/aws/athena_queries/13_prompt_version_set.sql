-- name: prompt_version_set
-- engine: athena (Presto/Trino SQL), runs on Athena or offline on DuckDB over Parquet
--         (same rule as qa/data_checks/checks/13_prompt_version_set.sql)
-- severity: medium
-- protects: Experiment tracking. Without prompt_version we cannot compare v1 vs v2 in production.
SELECT m.id AS row_id, m.role
FROM messages m
WHERE m.prompt_version IS NULL OR trim(m.prompt_version) = ''
