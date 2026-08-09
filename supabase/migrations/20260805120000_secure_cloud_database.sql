-- Keep application data behind the FastAPI server.
-- The browser authenticates with Clerk and calls the API; only the API uses
-- the backend-only Supabase service role key.

ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.project_settings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.project_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chats ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.messages ENABLE ROW LEVEL SECURITY;

REVOKE ALL PRIVILEGES
ON TABLE
    public.users,
    public.projects,
    public.project_settings,
    public.project_documents,
    public.document_chunks,
    public.chats,
    public.messages
FROM anon, authenticated;

GRANT SELECT, INSERT, UPDATE, DELETE
ON TABLE
    public.users,
    public.projects,
    public.project_settings,
    public.project_documents,
    public.document_chunks,
    public.chats,
    public.messages
TO service_role;

REVOKE EXECUTE
ON FUNCTION public.vector_search_document_chunks(
    vector,
    uuid[],
    double precision,
    integer
)
FROM PUBLIC, anon, authenticated;

REVOKE EXECUTE
ON FUNCTION public.keyword_search_document_chunks(text, uuid[], integer)
FROM PUBLIC, anon, authenticated;

GRANT EXECUTE
ON FUNCTION public.vector_search_document_chunks(
    vector,
    uuid[],
    double precision,
    integer
)
TO service_role;

GRANT EXECUTE
ON FUNCTION public.keyword_search_document_chunks(text, uuid[], integer)
TO service_role;

-- Apply the same backend-only boundary to future application objects.
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
REVOKE ALL ON TABLES FROM anon, authenticated;

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
REVOKE ALL ON SEQUENCES FROM anon, authenticated;

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC, anon, authenticated;

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO service_role;

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
GRANT USAGE, SELECT ON SEQUENCES TO service_role;

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
GRANT EXECUTE ON FUNCTIONS TO service_role;
