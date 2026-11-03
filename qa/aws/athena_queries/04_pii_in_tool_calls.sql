-- name: pii_in_tool_calls
-- engine: athena (Presto/Trino SQL), runs on Athena or offline on DuckDB over Parquet
--         (same rule as qa/data_checks/checks/04_pii_in_tool_calls.sql)
-- severity: critical
-- protects: Data protection. Tool arguments/results/errors (e.g. the phone given to
--           create_service_ticket) are logged too, so they must be masked as well.
SELECT tc.id AS row_id, tc.message_id, tc.name
FROM tool_calls tc
WHERE regexp_like(coalesce(tc.args_json, '') || ' ' || coalesce(tc.result_json, '') || ' ' || coalesce(tc.error, ''), '{{PHONE_RE}}')
   OR regexp_like(coalesce(tc.args_json, '') || ' ' || coalesce(tc.result_json, '') || ' ' || coalesce(tc.error, ''), '{{AADHAAR_RE}}')
   OR regexp_like(replace(lower(coalesce(tc.args_json, '') || ' ' || coalesce(tc.result_json, '') || ' ' || coalesce(tc.error, '')), '@hawahome.example', ''), '{{EMAIL_RE}}')
