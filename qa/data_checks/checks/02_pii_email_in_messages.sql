-- name: pii_email_in_messages
-- severity: critical
-- protects: Data protection. No customer email addresses in chat logs.
--           The official support address (@hawahome.example) is allowed, so it is removed first.
SELECT m.id AS row_id, m.conversation_id, m.role
FROM messages m
WHERE regexp_like(replace(lower(m.content), '@hawahome.example', ''), '{{EMAIL_RE}}')
