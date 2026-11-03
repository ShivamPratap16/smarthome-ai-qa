-- name: prompt_version_set
-- severity: medium
-- protects: Experiment tracking. Without prompt_version we cannot compare v1 vs v2 in production.
SELECT m.id AS row_id, m.role
FROM messages m
WHERE m.prompt_version IS NULL OR trim(m.prompt_version) = ''
