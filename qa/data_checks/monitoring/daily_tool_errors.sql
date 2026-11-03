-- name: daily_tool_errors
-- what: per day - number of device tool calls and how many of them returned an error (404/409/503...).
SELECT substr(m.created_at, 1, 10) AS day,
       count(*) AS tool_calls,
       sum(CASE WHEN tc.error IS NOT NULL THEN 1 ELSE 0 END) AS tool_errors
FROM tool_calls tc
JOIN messages m ON m.id = tc.message_id
GROUP BY substr(m.created_at, 1, 10)
ORDER BY day
