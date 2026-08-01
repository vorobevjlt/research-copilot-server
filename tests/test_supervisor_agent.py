import unittest
from unittest.mock import Mock, call, patch

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agents.supervisor_agent import agent as supervisor_agent


class ToolCallingFakeChatModel(FakeMessagesListChatModel):
    """Fake chat model that supports LangChain tool binding."""

    def bind_tools(self, tools, **kwargs):
        return self


class SupervisorPromptAndToolTests(unittest.TestCase):
    def test_prompt_routes_matching_saved_websites_to_scrapingbee(self):
        prompt = supervisor_agent.get_supervisor_system_prompt(
            website_sources=[
                {"source_url": "https://rentals.example/listings"}
            ]
        ).lower()

        self.assertIn("only `rag_search`", prompt)
        self.assertIn("`scrape_project_website`", prompt)
        self.assertIn("only `search_web`", prompt)
        self.assertIn("must call `scrape_project_website`", prompt)
        self.assertIn("prefer `scrape_project_website`", prompt)
        self.assertIn("https://rentals.example/listings", prompt)
        self.assertIn("use neither tool", prompt)
        self.assertIn("do not call a tool merely because it is available", prompt)
        self.assertIn("separately for each topic", prompt)
        self.assertIn("one topic cannot crowd the others out", prompt)
        self.assertIn("invoke at least one evidence tool before answering", prompt)
        self.assertIn("requires three focused `rag_search` calls", prompt)
        self.assertIn("rendered by the user interface after the answer", prompt)
        self.assertIn("do not add a duplicate textual `sources` section", prompt)

    def test_supervisor_exposes_exactly_three_tools(self):
        with patch.object(
            supervisor_agent, "_create_web_search_backend", return_value=None
        ):
            tools = supervisor_agent.create_supervisor_tools(
                "project-id",
                website_sources=[],
            )

        self.assertEqual(
            [tool.name for tool in tools],
            ["rag_search", "scrape_project_website", "search_web"],
        )

    def test_supervisor_appends_scrapingbee_mcp_tools(self):
        fast_search = Mock()
        fast_search.name = "fast_search"
        page_text = Mock()
        page_text.name = "get_page_text"

        with patch.object(
            supervisor_agent, "_create_web_search_backend", return_value=None
        ):
            tools = supervisor_agent.create_supervisor_tools(
                "project-id",
                website_sources=[],
                mcp_tools=[fast_search, page_text],
            )

        self.assertEqual(
            [tool.name for tool in tools],
            [
                "rag_search",
                "scrape_project_website",
                "search_web",
                "fast_search",
                "get_page_text",
            ],
        )

    def test_prompt_identifies_bound_scrapingbee_mcp_tools(self):
        fast_search = Mock()
        fast_search.name = "fast_search"
        page_text = Mock()
        page_text.name = "get_page_text"

        prompt = supervisor_agent.get_supervisor_system_prompt(
            website_sources=[],
            mcp_tools=[fast_search, page_text],
        ).lower()

        self.assertIn("scrapingbee mcp tools available", prompt)
        self.assertIn("fast_search", prompt)
        self.assertIn("get_page_text", prompt)
        self.assertIn("use `fast_search` first", prompt)

    def test_scrapingbee_tool_fetches_only_the_matching_saved_website(self):
        response = Mock()
        response.text = "<html><body><h1>Apartment A</h1></body></html>"
        saved_source = {
            "id": "website-document-1",
            "filename": "https://rentals.example/listings",
            "source_url": "https://rentals.example/listings",
        }

        with patch.object(
            supervisor_agent.scrapingbee_client,
            "html_api",
            return_value=response,
        ) as scrape:
            scrape_tool = supervisor_agent.create_project_website_scraping_tool(
                "project-id",
                website_sources=[saved_source],
            )
            result = scrape_tool.func(
                url="rentals.example",
                tool_call_id="call-scrape-1",
            )

        scrape.assert_called_once_with(
            "https://rentals.example/listings",
            params={"render_js": True},
        )
        response.raise_for_status.assert_called_once_with()
        self.assertIn("Apartment A", result.update["messages"][-1].content)
        self.assertEqual(
            result.update["citations"],
            [
                {
                    "chunk_id": None,
                    "document_id": "website-document-1",
                    "filename": "https://rentals.example/listings",
                    "page": "Live",
                    "source_type": "project_website",
                    "title": "https://rentals.example/listings",
                    "url": "https://rentals.example/listings",
                }
            ],
        )
        self.assertEqual(
            result.update["messages"][-1].artifact["citations"],
            result.update["citations"],
        )

    def test_scrapingbee_tool_rejects_a_website_not_saved_in_project(self):
        with patch.object(
            supervisor_agent.scrapingbee_client,
            "html_api",
        ) as scrape:
            scrape_tool = supervisor_agent.create_project_website_scraping_tool(
                "project-id",
                website_sources=[
                    {
                        "id": "website-document-1",
                        "filename": "https://rentals.example/listings",
                        "source_url": "https://rentals.example/listings",
                    }
                ],
            )
            result = scrape_tool.func(
                url="https://unrelated.example/listings",
                tool_call_id="call-scrape-2",
            )

        scrape.assert_not_called()
        self.assertIn(
            "does not uniquely match",
            result.update["messages"][-1].content,
        )
        self.assertNotIn("citations", result.update)

    def test_web_tool_returns_backend_results(self):
        backend = Mock()
        backend.invoke.return_value = {
            "results": [{"title": "Result", "url": "https://example.com"}]
        }

        with patch.object(
            supervisor_agent,
            "_create_web_search_backend",
            return_value=backend,
        ):
            web_tool = supervisor_agent.create_web_search_tool()
            result = web_tool.func(
                query="latest project news",
                tool_call_id="call-web-1",
            )

        backend.invoke.assert_called_once_with(
            {"query": "latest project news"}
        )
        self.assertIn(
            "https://example.com",
            result.update["messages"][-1].content,
        )
        self.assertEqual(
            result.update["citations"],
            [
                {
                    "chunk_id": None,
                    "document_id": "https://example.com",
                    "filename": "Result",
                    "page": "Web",
                    "source_type": "web",
                    "title": "Result",
                    "url": "https://example.com",
                }
            ],
        )
        self.assertEqual(
            result.update["messages"][-1].artifact["citations"],
            result.update["citations"],
        )

    def test_rag_tool_propagates_citations(self):
        citations = [
            {
                "document_id": "doc-1",
                "page": 2,
                "filename": "guide.pdf",
            }
        ]

        with (
            patch.object(
                supervisor_agent,
                "retrieve_context",
                return_value=(["context"], [], [], citations),
            ),
            patch.object(
                supervisor_agent,
                "prepare_prompt_and_invoke_llm",
                return_value="Grounded answer",
            ),
        ):
            rag_tool = supervisor_agent.create_rag_tool("project-id")
            result = rag_tool.func(
                query="What does the guide say?",
                tool_call_id="call-1",
            )

        self.assertEqual(result.update["citations"], citations)
        self.assertEqual(
            result.update["messages"][-1].artifact["citations"],
            citations,
        )
        self.assertEqual(
            result.update["messages"][-1].content,
            "Grounded answer",
        )


class SupervisorInputGuardrailTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "messages": [HumanMessage(content="Summarize the roadmap.")]
        }

    def test_safe_input_passes_toxic_and_injection_guards(self):
        toxic_guard = Mock()
        injection_guard = Mock()
        reporter = Mock()

        with (
            patch.object(
                supervisor_agent, "input_toxic_guard", toxic_guard
            ),
            patch.object(
                supervisor_agent,
                "input_injection_guard",
                injection_guard,
            ),
            patch.object(
                supervisor_agent, "_report_guardrail_result", reporter
            ),
        ):
            result = supervisor_agent.input_guardrail_node(self.state)

        toxic_guard.validate.assert_called_once_with("Summarize the roadmap.")
        injection_guard.validate.assert_called_once_with(
            "Summarize the roadmap."
        )
        self.assertEqual(
            reporter.call_args_list,
            [
                call("Toxic Content", "PASSED"),
                call("Prompt Injection", "PASSED"),
            ],
        )
        self.assertTrue(result["input_safe"])

    def test_prompt_injection_is_blocked_before_supervisor(self):
        toxic_guard = Mock()
        injection_guard = Mock()
        injection_guard.validate.side_effect = RuntimeError(
            "injection detected"
        )
        reporter = Mock()
        tool_reporter = Mock()

        with (
            patch.object(
                supervisor_agent, "input_toxic_guard", toxic_guard
            ),
            patch.object(
                supervisor_agent,
                "input_injection_guard",
                injection_guard,
            ),
            patch.object(
                supervisor_agent, "_report_guardrail_result", reporter
            ),
            patch.object(
                supervisor_agent, "_report_tool_usage", tool_reporter
            ),
        ):
            result = supervisor_agent.input_guardrail_node(self.state)

        tool_reporter.assert_called_once_with([])
        self.assertFalse(result["input_safe"])
        self.assertFalse(result["output_safe"])
        self.assertEqual(
            reporter.call_args_list,
            [
                call("Toxic Content", "PASSED"),
                call("Prompt Injection", "BLOCKED"),
                call("PII", "SKIPPED"),
            ],
        )

    def test_toxic_input_skips_remaining_guards(self):
        toxic_guard = Mock()
        toxic_guard.validate.side_effect = RuntimeError("toxic")
        injection_guard = Mock()
        reporter = Mock()
        tool_reporter = Mock()

        with (
            patch.object(
                supervisor_agent, "input_toxic_guard", toxic_guard
            ),
            patch.object(
                supervisor_agent,
                "input_injection_guard",
                injection_guard,
            ),
            patch.object(
                supervisor_agent, "_report_guardrail_result", reporter
            ),
            patch.object(
                supervisor_agent, "_report_tool_usage", tool_reporter
            ),
        ):
            result = supervisor_agent.input_guardrail_node(self.state)

        tool_reporter.assert_called_once_with([])
        self.assertFalse(result["input_safe"])
        injection_guard.validate.assert_not_called()
        self.assertEqual(
            reporter.call_args_list,
            [
                call("Toxic Content", "BLOCKED"),
                call("Prompt Injection", "SKIPPED"),
                call("PII", "SKIPPED"),
            ],
        )


class SupervisorOutputGuardrailTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "messages": [HumanMessage(content="A safe final answer.")],
            "input_safe": True,
        }

    def test_safe_output_passes_pii_guard(self):
        pii_guard = Mock()
        reporter = Mock()

        with (
            patch.object(supervisor_agent, "output_guard", pii_guard),
            patch.object(
                supervisor_agent, "_report_guardrail_result", reporter
            ),
        ):
            result = supervisor_agent.output_guardrail_node(self.state)

        pii_guard.validate.assert_called_once_with("A safe final answer.")
        reporter.assert_called_once_with("PII", "PASSED")
        self.assertTrue(result["output_safe"])
        self.assertEqual(result["final_response"], "A safe final answer.")

    def test_pii_output_is_replaced(self):
        pii_guard = Mock()
        pii_guard.validate.side_effect = RuntimeError("PII detected")
        reporter = Mock()

        with (
            patch.object(supervisor_agent, "output_guard", pii_guard),
            patch.object(
                supervisor_agent, "_report_guardrail_result", reporter
            ),
        ):
            result = supervisor_agent.output_guardrail_node(self.state)

        reporter.assert_called_once_with("PII", "BLOCKED")
        self.assertFalse(result["output_safe"])
        self.assertNotEqual(
            result["messages"][-1].content,
            "A safe final answer.",
        )


class SupervisorSourceIntegrationTests(unittest.TestCase):
    def test_multi_topic_rag_sources_are_combined_for_the_ui(self):
        model = ToolCallingFakeChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "rag_search",
                            "args": {"query": "scaled dot-product attention formula"},
                            "id": "attention-call",
                            "type": "tool_call",
                        },
                        {
                            "name": "rag_search",
                            "args": {
                                "query": (
                                    "neural network layers weights learning "
                                    "convolution classification"
                                )
                            },
                            "id": "network-call",
                            "type": "tool_call",
                        },
                        {
                            "name": "rag_search",
                            "args": {"query": "human long-term memory brain"},
                            "id": "memory-call",
                            "type": "tool_call",
                        },
                    ],
                ),
                AIMessage(content="A combined three-part answer."),
            ]
        )
        citations_by_query = {
            "scaled dot-product attention formula": [
                {
                    "document_id": "attention-doc",
                    "filename": "attention.pdf",
                    "page": 3,
                }
            ],
            "neural network layers weights learning convolution classification": [
                {
                    "document_id": "network-doc",
                    "filename": "neural-networks.pdf",
                    "page": 4,
                }
            ],
            "human long-term memory brain": [
                {
                    "document_id": "memory-doc",
                    "filename": "neuroscience.txt",
                    "page": 6,
                }
            ],
        }

        def retrieve_for_topic(_project_id, query):
            return (["context"], [], [], citations_by_query[query])

        with (
            patch.object(supervisor_agent, "input_toxic_guard", Mock()),
            patch.object(
                supervisor_agent, "input_injection_guard", Mock()
            ),
            patch.object(supervisor_agent, "output_guard", Mock()),
            patch.object(
                supervisor_agent,
                "_create_web_search_backend",
                return_value=None,
            ),
            patch.object(
                supervisor_agent,
                "retrieve_context",
                side_effect=retrieve_for_topic,
            ),
            patch.object(
                supervisor_agent,
                "prepare_prompt_and_invoke_llm",
                return_value="Grounded topic answer",
            ),
            patch.object(supervisor_agent, "_report_guardrail_result"),
            patch.object(supervisor_agent, "_report_tool_usage"),
        ):
            agent = supervisor_agent.create_supervisor_agent(
                "project-id", model=model, website_sources=[]
            )
            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Give me the attention formula, neural-network "
                                "idea, and human long-term memory."
                            ),
                        }
                    ]
                }
            )

        self.assertEqual(
            [citation["filename"] for citation in result["citations"]],
            ["attention.pdf", "neural-networks.pdf", "neuroscience.txt"],
        )

    def test_rag_sources_survive_after_the_final_answer(self):
        model = ToolCallingFakeChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "rag_search",
                            "args": {"query": "roadmap"},
                            "id": "rag-call",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="The roadmap has three stages."),
            ]
        )
        citations = [
            {
                "document_id": "doc-1",
                "filename": "roadmap.pdf",
                "page": 2,
            }
        ]

        with (
            patch.object(supervisor_agent, "input_toxic_guard", Mock()),
            patch.object(
                supervisor_agent, "input_injection_guard", Mock()
            ),
            patch.object(supervisor_agent, "output_guard", Mock()),
            patch.object(
                supervisor_agent,
                "_create_web_search_backend",
                return_value=None,
            ),
            patch.object(
                supervisor_agent,
                "retrieve_context",
                return_value=(["context"], [], [], citations),
            ),
            patch.object(
                supervisor_agent,
                "prepare_prompt_and_invoke_llm",
                return_value="Grounded evidence",
            ),
            patch.object(supervisor_agent, "_report_guardrail_result"),
            patch.object(supervisor_agent, "_report_tool_usage"),
        ):
            agent = supervisor_agent.create_supervisor_agent(
                "project-id", model=model, website_sources=[]
            )
            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is the roadmap?",
                        }
                    ]
                }
            )

        self.assertEqual(
            result["messages"][-1].content,
            "The roadmap has three stages.",
        )
        self.assertEqual(result["citations"], citations)

    def test_web_sources_survive_after_the_final_answer(self):
        model = ToolCallingFakeChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_web",
                            "args": {"query": "latest roadmap news"},
                            "id": "web-call",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="The latest public update is available."),
            ]
        )
        backend = Mock()
        backend.invoke.return_value = {
            "results": [
                {
                    "title": "Public roadmap update",
                    "url": "https://example.com/roadmap",
                }
            ]
        }

        with (
            patch.object(supervisor_agent, "input_toxic_guard", Mock()),
            patch.object(
                supervisor_agent, "input_injection_guard", Mock()
            ),
            patch.object(supervisor_agent, "output_guard", Mock()),
            patch.object(
                supervisor_agent,
                "_create_web_search_backend",
                return_value=backend,
            ),
            patch.object(supervisor_agent, "_report_guardrail_result"),
            patch.object(supervisor_agent, "_report_tool_usage"),
        ):
            agent = supervisor_agent.create_supervisor_agent(
                "project-id", model=model, website_sources=[]
            )
            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is the latest roadmap news?",
                        }
                    ]
                }
            )

        self.assertEqual(
            result["messages"][-1].content,
            "The latest public update is available.",
        )
        self.assertEqual(
            result["citations"][0]["url"],
            "https://example.com/roadmap",
        )

    def test_saved_website_scrape_sources_survive_after_the_final_answer(self):
        saved_source = {
            "id": "website-document-1",
            "filename": "https://rentals.example/listings",
            "source_url": "https://rentals.example/listings",
        }
        model = ToolCallingFakeChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "scrape_project_website",
                            "args": {
                                "url": "https://rentals.example/listings"
                            },
                            "id": "scrape-call",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="The saved website has a current listing."),
            ]
        )
        response = Mock()
        response.text = "<html><body>Current apartment listing</body></html>"

        with (
            patch.object(supervisor_agent, "input_toxic_guard", Mock()),
            patch.object(
                supervisor_agent, "input_injection_guard", Mock()
            ),
            patch.object(supervisor_agent, "output_guard", Mock()),
            patch.object(
                supervisor_agent,
                "_create_web_search_backend",
                return_value=None,
            ),
            patch.object(
                supervisor_agent.scrapingbee_client,
                "html_api",
                return_value=response,
            ) as scrape,
            patch.object(supervisor_agent, "_report_guardrail_result"),
            patch.object(supervisor_agent, "_report_tool_usage"),
        ):
            agent = supervisor_agent.create_supervisor_agent(
                "project-id",
                model=model,
                website_sources=[saved_source],
            )
            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Find a current apartment listing.",
                        }
                    ]
                }
            )

        scrape.assert_called_once_with(
            "https://rentals.example/listings",
            params={"render_js": True},
        )
        self.assertEqual(
            result["messages"][-1].content,
            "The saved website has a current listing.",
        )
        self.assertEqual(result["citations"][0]["source_type"], "project_website")
        self.assertEqual(
            result["citations"][0]["url"],
            "https://rentals.example/listings",
        )


class SupervisorCitationTests(unittest.TestCase):
    def test_citations_are_deduplicated_by_document_and_page(self):
        current = [{"document_id": "doc-1", "page": 1}]
        incoming = [
            {"document_id": "doc-1", "page": 1},
            {"document_id": "doc-1", "page": 2},
        ]

        self.assertEqual(
            supervisor_agent.merge_citations(current, incoming),
            [
                {"document_id": "doc-1", "page": 1},
                {"document_id": "doc-1", "page": 2},
            ],
        )

    def test_web_citations_are_deduplicated_by_url(self):
        current = [
            {
                "source_type": "web",
                "url": "https://example.com/a",
                "filename": "Example A",
            }
        ]
        incoming = [
            {
                "source_type": "web",
                "url": "https://example.com/a",
                "filename": "Duplicate",
            },
            {
                "source_type": "web",
                "url": "https://example.com/b",
                "filename": "Example B",
            },
        ]

        self.assertEqual(
            supervisor_agent.merge_citations(current, incoming),
            [
                current[0],
                incoming[1],
            ],
        )


class SupervisorToolUsageTests(unittest.TestCase):
    def test_reports_tools_in_invocation_order(self):
        messages = [
            HumanMessage(content="Compare internal and external information."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "rag_search",
                        "args": {"query": "internal"},
                        "id": "rag-call",
                        "type": "tool_call",
                    },
                    {
                        "name": "search_web",
                        "args": {"query": "external"},
                        "id": "web-call",
                        "type": "tool_call",
                    },
                ],
            ),
        ]
        reporter = Mock()

        with patch.object(
            supervisor_agent, "_report_tool_usage", reporter
        ):
            result = supervisor_agent.tool_usage_node(
                {"messages": messages}
            )

        reporter.assert_called_once_with(["rag_search", "search_web"])
        self.assertEqual(
            result["tools_used"],
            ["rag_search", "search_web"],
        )

    def test_recovers_sources_from_tool_messages(self):
        citations = [
            {
                "document_id": "doc-1",
                "filename": "roadmap.pdf",
                "page": 2,
            }
        ]
        messages = [
            HumanMessage(content="What is the roadmap?"),
            ToolMessage(
                content="Grounded evidence",
                tool_call_id="rag-call",
                artifact={"citations": citations},
            ),
            AIMessage(content="The roadmap has three stages."),
        ]

        with patch.object(supervisor_agent, "_report_tool_usage"):
            result = supervisor_agent.tool_usage_node(
                {"messages": messages}
            )

        self.assertEqual(result["citations"], citations)

    def test_recovers_scrapingbee_mcp_search_sources(self):
        messages = [
            HumanMessage(content="Find Berlin apartment listings."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "fast_search",
                        "args": {"search": "Berlin apartments"},
                        "id": "mcp-search-call",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="Search completed",
                name="fast_search",
                tool_call_id="mcp-search-call",
                artifact={
                    "structured_content": {
                        "organic": [
                            {
                                "title": "Berlin rentals",
                                "link": "https://rentals.example/berlin",
                            }
                        ]
                    }
                },
            ),
            AIMessage(content="I found one relevant rental source."),
        ]

        with patch.object(supervisor_agent, "_report_tool_usage"):
            result = supervisor_agent.tool_usage_node({"messages": messages})

        self.assertEqual(
            result["citations"],
            [
                {
                    "chunk_id": None,
                    "document_id": "https://rentals.example/berlin",
                    "filename": "Berlin rentals",
                    "page": "Web",
                    "source_type": "web",
                    "title": "Berlin rentals",
                    "url": "https://rentals.example/berlin",
                }
            ],
        )

    def test_recovers_scrapingbee_mcp_page_url_from_tool_arguments(self):
        source_url = "https://rentals.example/listing/123"
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_page_text",
                        "args": {"url": source_url},
                        "id": "mcp-page-call",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content="Apartment details",
                name="get_page_text",
                tool_call_id="mcp-page-call",
            ),
        ]

        with patch.object(supervisor_agent, "_report_tool_usage"):
            result = supervisor_agent.tool_usage_node({"messages": messages})

        self.assertEqual(result["citations"][0]["url"], source_url)
        self.assertEqual(result["citations"][0]["source_type"], "web")

    def test_reports_none_when_no_tool_was_needed(self):
        reporter = Mock()

        with patch.object(
            supervisor_agent, "_report_tool_usage", reporter
        ):
            result = supervisor_agent.tool_usage_node(
                {
                    "messages": [
                        HumanMessage(content="Hello"),
                        AIMessage(content="Hello!"),
                    ]
                }
            )

        reporter.assert_called_once_with([])
        self.assertEqual(result["tools_used"], [])
        self.assertEqual(result["citations"], [])


if __name__ == "__main__":
    unittest.main()
