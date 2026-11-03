-- name: pii_in_tickets
-- severity: critical
-- protects: Data protection. Service tickets keep only a masked phone (no 4+ digit run left)
--           and the issue text must be clean too.
SELECT t.id AS row_id, t.device_id
FROM tickets t
WHERE regexp_like(coalesce(t.issue, '') || ' ' || coalesce(t.phone_masked, ''), '{{PHONE_RE}}')
   OR regexp_like(coalesce(t.issue, '') || ' ' || coalesce(t.phone_masked, ''), '{{AADHAAR_RE}}')
   OR regexp_like(replace(lower(coalesce(t.issue, '')), '@hawahome.example', ''), '{{EMAIL_RE}}')
   OR regexp_like(coalesce(t.phone_masked, ''), '[0-9]{4}')
