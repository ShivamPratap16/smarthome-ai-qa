-- name: daily_topics
-- what: per day - which product the USER asked about (simple keyword rules). Used for topic drift.
SELECT day, topic, count(*) AS n
FROM (
    SELECT substr(m.created_at, 1, 10) AS day,
           CASE
               WHEN lower(m.content) LIKE '%warranty%' THEN 'warranty'
               WHEN lower(m.content) LIKE '%ticket%' OR lower(m.content) LIKE '%technician%' THEN 'ticket'
               WHEN lower(m.content) LIKE '%fan%' OR lower(m.content) LIKE '%aero%'
                    OR lower(m.content) LIKE '%pankha%' THEN 'aero'
               WHEN lower(m.content) LIKE '%lock%' OR lower(m.content) LIKE '%door%' THEN 'lock'
               WHEN lower(m.content) LIKE '%purifier%' OR lower(m.content) LIKE '%water%'
                    OR lower(m.content) LIKE '%filter%' OR lower(m.content) LIKE '%tds%' THEN 'pure'
               ELSE 'other'
           END AS topic
    FROM messages m
    WHERE m.role = 'user'
) t
GROUP BY day, topic
ORDER BY day, topic
