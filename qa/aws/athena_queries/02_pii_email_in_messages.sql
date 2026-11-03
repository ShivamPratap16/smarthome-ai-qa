-- name: pii_email_in_messages
-- engine: athena (Presto/Trino SQL), runs on Athena or offline on DuckDB over Parquet
--         (same rule as qa/data_checks/checks/02_pii_email_in_messages.sql)
-- severity: critical
-- protects: Data protection. No customer email addresses in chat logs.
--           The official support address (@hawahome.example) is allowed, so it is removed first.
SELECT m.id AS row_id, m.conversation_id, m.role
FROM messages m
WHERE regexp_like(replace(lower(m.content), '@hawahome.example', ''), '{{EMAIL_RE}}')
