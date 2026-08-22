-- Runs once, when the Postgres data directory is first created.
--
-- Separate databases rather than separate schemas: LiteLLM and Langfuse both
-- run their own migrations and expect to own their database. Sharing one would
-- mean their migration histories collide.
--
-- Adding a database later means either running this by hand or destroying the
-- volume, because docker-entrypoint-initdb.d is ignored once the data
-- directory exists. Hence creating both up front, including the one that is
-- not needed until Langfuse is self-hosted.

CREATE DATABASE litellm;
CREATE DATABASE langfuse;

COMMENT ON DATABASE litellm IS 'LiteLLM proxy: virtual keys, budgets, spend';
COMMENT ON DATABASE langfuse IS 'Langfuse, if self-hosted later';
