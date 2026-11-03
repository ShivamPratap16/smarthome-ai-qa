-- name: no_orphan_tickets
-- severity: high
-- protects: Traceability. Every ticket must come from a logged create_service_ticket tool call
--           (a ticket with no tool call = created outside the bot, or a lost log row).
SELECT t.id AS row_id, t.device_id
FROM tickets t
WHERE NOT EXISTS (
    SELECT 1 FROM tool_calls tc
    WHERE tc.name = 'create_service_ticket'
      AND tc.result_json LIKE '%' || t.id || '%')
