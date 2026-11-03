-- name: message_has_conversation
-- severity: high
-- protects: Referential integrity. No message without its conversation row.
SELECT m.id AS row_id, m.conversation_id
FROM messages m
LEFT JOIN conversations c ON c.id = m.conversation_id
WHERE c.id IS NULL
