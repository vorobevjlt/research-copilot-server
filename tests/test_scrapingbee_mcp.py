import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.services import scrapingbee_mcp


class ScrapingBeeMCPServiceTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        scrapingbee_mcp.clear_scrapingbee_mcp_tools_cache()

    def test_endpoint_url_encodes_the_api_key(self):
        endpoint = scrapingbee_mcp._build_scrapingbee_mcp_url("key+with/slash")

        self.assertEqual(
            endpoint,
            "https://mcp.scrapingbee.com/mcp?api_key=key%2Bwith%2Fslash",
        )

    def test_endpoint_rejects_an_empty_api_key(self):
        with self.assertRaises(
            scrapingbee_mcp.ScrapingBeeMCPConfigurationError
        ):
            scrapingbee_mcp._build_scrapingbee_mcp_url("")

    def test_error_redaction_handles_url_encoded_keys(self):
        with patch.dict(
            scrapingbee_mcp.appConfig,
            {"scrapingbee_api_key": "key+with/slash"},
        ):
            message = scrapingbee_mcp._redact_mcp_error(
                RuntimeError(
                    "request failed: api_key=key%2Bwith%2Fslash"
                )
            )

        self.assertNotIn("key%2Bwith%2Fslash", message)
        self.assertIn("[REDACTED]", message)

    async def test_loads_only_allowlisted_remote_tools(self):
        fast_search = SimpleNamespace(name="fast_search")
        page_text = SimpleNamespace(name="get_page_text")
        extra_tool = SimpleNamespace(name="ask_chatgpt")
        client = SimpleNamespace(
            get_tools=AsyncMock(
                return_value=[fast_search, page_text, extra_tool]
            )
        )

        with patch.object(
            scrapingbee_mcp,
            "create_scrapingbee_mcp_client",
            return_value=client,
        ):
            tools = await scrapingbee_mcp.get_scrapingbee_mcp_tools(
                allowed_names=("fast_search", "get_page_text")
            )

        client.get_tools.assert_awaited_once_with(server_name="scrapingbee")
        self.assertEqual(
            [tool.name for tool in tools],
            ["fast_search", "get_page_text"],
        )

    async def test_remote_failure_returns_fallback_empty_tool_list(self):
        client = SimpleNamespace(
            get_tools=AsyncMock(side_effect=RuntimeError("offline"))
        )

        with patch.object(
            scrapingbee_mcp,
            "create_scrapingbee_mcp_client",
            return_value=client,
        ):
            tools = await scrapingbee_mcp.get_scrapingbee_mcp_tools()

        self.assertEqual(tools, [])


if __name__ == "__main__":
    unittest.main()
