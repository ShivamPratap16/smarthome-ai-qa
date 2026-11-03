-- Athena / AWS Glue tables over the Parquet lake written by `python -m qa.aws.export`.
-- Run these once in the Athena console (replace YOUR-BUCKET). They only describe where the
-- files are; Athena never copies the data. "Partition projection" lets Athena work out the
-- dt=YYYY-MM-DD folders by itself, so no MSCK REPAIR TABLE / crawler is needed.

CREATE DATABASE IF NOT EXISTS hawa_chatlogs;

CREATE EXTERNAL TABLE IF NOT EXISTS hawa_chatlogs.conversations (
  id string, user_id string, started_at string)
PARTITIONED BY (dt string)
STORED AS PARQUET
LOCATION 's3://YOUR-BUCKET/chatlogs/conversations/'
TBLPROPERTIES ('projection.enabled'='true', 'projection.dt.type'='date',
  'projection.dt.format'='yyyy-MM-dd', 'projection.dt.range'='2026-01-01,NOW',
  'storage.location.template'='s3://YOUR-BUCKET/chatlogs/conversations/dt=${dt}/');

CREATE EXTERNAL TABLE IF NOT EXISTS hawa_chatlogs.messages (
  id bigint, conversation_id string, role string, content string, created_at string,
  prompt_version string, latency_ms bigint, tokens_in bigint, tokens_out bigint,
  cost_usd double, refused int, blocked_by string)
PARTITIONED BY (dt string)
STORED AS PARQUET
LOCATION 's3://YOUR-BUCKET/chatlogs/messages/'
TBLPROPERTIES ('projection.enabled'='true', 'projection.dt.type'='date',
  'projection.dt.format'='yyyy-MM-dd', 'projection.dt.range'='2026-01-01,NOW',
  'storage.location.template'='s3://YOUR-BUCKET/chatlogs/messages/dt=${dt}/');

CREATE EXTERNAL TABLE IF NOT EXISTS hawa_chatlogs.tool_calls (
  id bigint, message_id bigint, name string, args_json string, result_json string, error string)
PARTITIONED BY (dt string)
STORED AS PARQUET
LOCATION 's3://YOUR-BUCKET/chatlogs/tool_calls/'
TBLPROPERTIES ('projection.enabled'='true', 'projection.dt.type'='date',
  'projection.dt.format'='yyyy-MM-dd', 'projection.dt.range'='2026-01-01,NOW',
  'storage.location.template'='s3://YOUR-BUCKET/chatlogs/tool_calls/dt=${dt}/');

CREATE EXTERNAL TABLE IF NOT EXISTS hawa_chatlogs.tickets (
  id string, device_id string, issue string, phone_masked string, created_at string)
PARTITIONED BY (dt string)
STORED AS PARQUET
LOCATION 's3://YOUR-BUCKET/chatlogs/tickets/'
TBLPROPERTIES ('projection.enabled'='true', 'projection.dt.type'='date',
  'projection.dt.format'='yyyy-MM-dd', 'projection.dt.range'='2026-01-01,NOW',
  'storage.location.template'='s3://YOUR-BUCKET/chatlogs/tickets/dt=${dt}/');
