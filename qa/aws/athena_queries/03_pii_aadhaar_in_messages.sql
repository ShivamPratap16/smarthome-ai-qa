-- name: pii_aadhaar_in_messages
-- engine: athena (Presto/Trino SQL), runs on Athena or offline on DuckDB over Parquet
--         (same rule as qa/data_checks/checks/03_pii_aadhaar_in_messages.sql)
-- severity: critical
-- protects: Data protection. No 12-digit Aadhaar-like identity numbers in chat logs.
SELECT m.id AS row_id, m.conversation_id, m.role
FROM messages m
WHERE regexp_like(m.content, '{{AADHAAR_RE}}')
