-- The API server connects to Supabase with the backend-only service_role key.
-- Tables created by migrations do not automatically grant that role data access.

GRANT USAGE ON SCHEMA public TO service_role;

GRANT SELECT, INSERT, UPDATE, DELETE
ON ALL TABLES IN SCHEMA public
TO service_role;

GRANT USAGE, SELECT
ON ALL SEQUENCES IN SCHEMA public
TO service_role;

GRANT EXECUTE
ON ALL FUNCTIONS IN SCHEMA public
TO service_role;

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO service_role;

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
GRANT USAGE, SELECT ON SEQUENCES TO service_role;

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
GRANT EXECUTE ON FUNCTIONS TO service_role;
