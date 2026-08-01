"""ScrapingBee Remote MCP tools exposed to the Agent_01 supervisor."""

import asyncio
import logging
from typing import Iterable, Optional
from urllib.parse import urlencode

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.config.index import appConfig


logger = logging.getLogger(__name__)

SCRAPINGBEE_MCP_SERVER_NAME = "scrapingbee"
SCRAPINGBEE_MCP_BASE_URL = "https://mcp.scrapingbee.com/mcp"

# Keep the supervisor focused on public-web research and listing extraction.
# Additional ScrapingBee MCP tools can be added here when the product needs them.
SCRAPINGBEE_MCP_TOOL_ALLOWLIST = (
    "fast_search",
    "get_page_text",
    "extract_page_data",
    "get_screenshot",
)


class ScrapingBeeMCPConfigurationError(RuntimeError):
    """Raised when the ScrapingBee MCP client cannot be configured."""


_tools_cache: Optional[tuple[BaseTool, ...]] = None
_tools_lock = asyncio.Lock()


def _build_scrapingbee_mcp_url(api_key: str) -> str:
    """Build the MCP endpoint without storing the secret in source control."""
    if not isinstance(api_key, str) or not api_key.strip():
        raise ScrapingBeeMCPConfigurationError(
            "SCRAPINGBEE_API_KEY is required for ScrapingBee MCP."
        )

    query = urlencode({"api_key": api_key.strip()})
    return f"{SCRAPINGBEE_MCP_BASE_URL}?{query}"


def create_scrapingbee_mcp_client(
    api_key: Optional[str] = None,
) -> MultiServerMCPClient:
    """Create a stateless Streamable HTTP client for ScrapingBee's MCP server."""
    resolved_key = api_key or appConfig.get("scrapingbee_api_key")
    endpoint = _build_scrapingbee_mcp_url(resolved_key)

    return MultiServerMCPClient(
        {
            SCRAPINGBEE_MCP_SERVER_NAME: {
                "transport": "streamable_http",
                "url": endpoint,
                "timeout": 30,
                "sse_read_timeout": 140,
                "terminate_on_close": True,
            }
        },
        handle_tool_errors=True,
    )


def _redact_mcp_error(error: Exception) -> str:
    """Return a log-safe error string that never includes the API key."""
    message = str(error)
    api_key = appConfig.get("scrapingbee_api_key")
    if isinstance(api_key, str) and api_key:
        message = message.replace(api_key, "[REDACTED]")
        encoded_key = urlencode({"api_key": api_key}).partition("=")[2]
        message = message.replace(encoded_key, "[REDACTED]")
    return message


async def get_scrapingbee_mcp_tools(
    allowed_names: Iterable[str] = SCRAPINGBEE_MCP_TOOL_ALLOWLIST,
) -> list[BaseTool]:
    """Load and cache the approved ScrapingBee MCP tools.

    If the remote MCP server is temporarily unavailable, Agent_01 keeps its
    existing RAG, saved-site scraping, and public-search tools and retries on a
    later request.
    """
    global _tools_cache

    allowed = tuple(dict.fromkeys(allowed_names))
    if _tools_cache is not None:
        cached_by_name = {tool.name: tool for tool in _tools_cache}
        return [
            cached_by_name[name]
            for name in allowed
            if name in cached_by_name
        ]

    async with _tools_lock:
        if _tools_cache is None:
            try:
                client = create_scrapingbee_mcp_client()
                loaded_tools = await client.get_tools(
                    server_name=SCRAPINGBEE_MCP_SERVER_NAME
                )
                _tools_cache = tuple(loaded_tools)
            except Exception as error:
                logger.warning(
                    "ScrapingBee MCP tools are unavailable; continuing with "
                    "Agent_01 fallback tools: %s",
                    _redact_mcp_error(error),
                )
                return []

    tools_by_name = {tool.name: tool for tool in _tools_cache}
    missing_names = [name for name in allowed if name not in tools_by_name]
    if missing_names:
        logger.warning(
            "ScrapingBee MCP did not advertise expected tools: %s",
            ", ".join(missing_names),
        )

    return [tools_by_name[name] for name in allowed if name in tools_by_name]


def clear_scrapingbee_mcp_tools_cache() -> None:
    """Clear cached tool definitions, primarily for tests and key rotation."""
    global _tools_cache
    _tools_cache = None
