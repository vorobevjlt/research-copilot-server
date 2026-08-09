## 09_Retrieval

### Setup

1. **Install System Dependencies:**

   These are required for document processing (PDFs, images, etc.)

   **macOS:**

   ```bash
   brew install poppler tesseract libmagic
   ```

   **Linux (Ubuntu/Debian):**

   ```bash
   sudo apt-get update
   sudo apt-get install poppler-utils tesseract-ocr libmagic1
   ```

2. **Install Python Dependencies:**

   ```bash
   poetry install
   ```

3. **Create a `.env` file:**

   ```bash
   cp .env.sample .env
   ```

   Then update the values in `.env` file with your configuration.

   Agentic projects can use ScrapingBee's Remote MCP server. Add:

   ```bash
   SCRAPINGBEE_API_KEY=your_api_key
   ```

   The supervisor loads these MCP tools lazily on the first agentic chat
   request: `fast_search`, `get_page_text`, `extract_page_data`, and
   `get_screenshot`. The API key is URL-encoded at runtime and is never stored
   in source-controlled MCP configuration. If the remote MCP server is
   unavailable, the supervisor keeps its existing RAG, saved-website scraper,
   and fallback web-search tools.

   > 💡 **Tip:** Get your Supabase credentials by running `npx supabase status` after starting Supabase locally.
   >
   > ⚠️ **Note:** Supabase has updated their naming. The old variable `service_role key` is now simply called `Secret Key`.
   > 📸 [Reference screenshot](https://ik.imagekit.io/5wegcvcxp/HarishNeel/supabase-credentials.png)

4. **Start All Services:**

   Build and start Redis, the API server, and the Celery worker with Docker
   Compose:

   ```bash
   docker compose up --build
   ```

   The server will run on `http://localhost:8000`.

5. **Stop All Services:**

   To stop everything at once:

   ```bash
   docker compose down
   ```

   This stops the Celery worker, Redis server, and API server containers.

### Summary

- Complete the Basic Retrieval Pipeline; every step is already well documented inside the code.
- Update the initial_schema by inserting changes before `(embedding vector_ip_ops);` and after `(embedding vector_cosine_ops);`
- Create a new migration file for the Postgres functions `vector_search_document_chunks` and `keyword_search_document_chunks`.
- Complete the Advanced Retrieval Pipeline.
