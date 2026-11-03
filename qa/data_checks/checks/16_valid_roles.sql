-- name: valid_roles
-- severity: medium
-- protects: Schema contract. Only 'user' and 'assistant' messages are logged.
SELECT m.id AS row_id, m.role
FROM messages m
WHERE m.role IS NULL OR m.role NOT IN ('user', 'assistant')
