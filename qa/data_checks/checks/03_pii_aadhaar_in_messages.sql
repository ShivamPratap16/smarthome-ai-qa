-- name: pii_aadhaar_in_messages
-- severity: critical
-- protects: Data protection. No 12-digit Aadhaar-like identity numbers in chat logs.
SELECT m.id AS row_id, m.conversation_id, m.role
FROM messages m
WHERE regexp_like(m.content, '{{AADHAAR_RE}}')
