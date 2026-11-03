-- name: pii_phone_in_messages
-- severity: critical
-- protects: Data protection. Chat logs must never store a raw Indian mobile number
--           (with or without +91 / leading 0 / spaces). mask_pii should have turned it into [PHONE].
-- returns: one row per violating message (ids only, never the content itself)
SELECT m.id AS row_id, m.conversation_id, m.role
FROM messages m
WHERE regexp_like(m.content, '{{PHONE_RE}}')
