-- name: ticket_id_unique
-- engine: athena (Presto/Trino SQL), runs on Athena or offline on DuckDB over Parquet
--         (same rule as qa/data_checks/checks/18_ticket_id_unique.sql)
-- severity: high
-- protects: Data integrity. One ticket id must come from exactly one create_service_ticket call.
--           If two calls returned the same id, the second ticket overwrote the first in the
--           tickets table and a customer's complaint was silently lost.
SELECT t.id AS row_id, count(*) AS tool_calls
FROM tickets t
JOIN tool_calls tc
  ON tc.name = 'create_service_ticket'
 AND tc.result_json LIKE '%"' || t.id || '"%'
GROUP BY t.id
HAVING count(*) > 1
