import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import HumanMessage

from src.agents.simple_agent import agent as simple_agent


class SimpleAgentInputGuardrailTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "messages": [HumanMessage(content="Summarize the project roadmap.")]
        }

    def test_safe_input_passes_both_input_guards(self):
        toxic_guard = Mock()
        injection_guard = Mock()
        reporter = Mock()

        with (
            patch.object(simple_agent, "input_guard", toxic_guard),
            patch.object(
                simple_agent, "input_injection_guard", injection_guard
            ),
            patch.object(
                simple_agent, "_report_guardrail_result", reporter
            ),
        ):
            result = simple_agent.input_guardrail_node(self.state)

        toxic_guard.validate.assert_called_once_with(
            "Summarize the project roadmap."
        )
        injection_guard.validate.assert_called_once_with(
            "Summarize the project roadmap."
        )
        self.assertEqual(
            reporter.call_args_list,
            [
                unittest.mock.call("Toxic Content", "PASSED"),
                unittest.mock.call("Prompt Injection", "PASSED"),
            ],
        )
        self.assertTrue(result["input_safe"])

    def test_prompt_injection_is_rejected_before_agent_execution(self):
        toxic_guard = Mock()
        injection_guard = Mock()
        reporter = Mock()
        injection_guard.validate.side_effect = RuntimeError(
            "prompt injection detected"
        )

        with (
            patch.object(simple_agent, "input_guard", toxic_guard),
            patch.object(
                simple_agent, "input_injection_guard", injection_guard
            ),
            patch.object(
                simple_agent, "_report_guardrail_result", reporter
            ),
        ):
            result = simple_agent.input_guardrail_node(self.state)

        self.assertEqual(
            reporter.call_args_list,
            [
                unittest.mock.call("Toxic Content", "PASSED"),
                unittest.mock.call("Prompt Injection", "BLOCKED"),
                unittest.mock.call("PII", "SKIPPED"),
            ],
        )
        self.assertFalse(result["input_safe"])
        self.assertFalse(result["output_safe"])
        self.assertIn("override or manipulate", result["final_response"])
        self.assertEqual(
            result["messages"][-1].content, result["final_response"]
        )

    def test_toxic_input_does_not_call_injection_detector(self):
        toxic_guard = Mock()
        toxic_guard.validate.side_effect = RuntimeError("toxic input")
        injection_guard = Mock()
        reporter = Mock()

        with (
            patch.object(simple_agent, "input_guard", toxic_guard),
            patch.object(
                simple_agent, "input_injection_guard", injection_guard
            ),
            patch.object(
                simple_agent, "_report_guardrail_result", reporter
            ),
        ):
            result = simple_agent.input_guardrail_node(self.state)

        self.assertFalse(result["input_safe"])
        injection_guard.validate.assert_not_called()
        self.assertEqual(
            reporter.call_args_list,
            [
                unittest.mock.call("Toxic Content", "BLOCKED"),
                unittest.mock.call("Prompt Injection", "SKIPPED"),
                unittest.mock.call("PII", "SKIPPED"),
            ],
        )


class SimpleAgentOutputGuardrailTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "messages": [HumanMessage(content="A safe generated answer.")],
            "input_safe": True,
        }

    def test_safe_output_reports_pii_passed(self):
        pii_guard = Mock()
        reporter = Mock()

        with (
            patch.object(simple_agent, "output_guard", pii_guard),
            patch.object(
                simple_agent, "_report_guardrail_result", reporter
            ),
        ):
            result = simple_agent.output_guardrail_node(self.state)

        pii_guard.validate.assert_called_once_with("A safe generated answer.")
        reporter.assert_called_once_with("PII", "PASSED")
        self.assertTrue(result["output_safe"])

    def test_pii_output_reports_blocked(self):
        pii_guard = Mock()
        pii_guard.validate.side_effect = RuntimeError("PII detected")
        reporter = Mock()

        with (
            patch.object(simple_agent, "output_guard", pii_guard),
            patch.object(
                simple_agent, "_report_guardrail_result", reporter
            ),
        ):
            result = simple_agent.output_guardrail_node(self.state)

        reporter.assert_called_once_with("PII", "BLOCKED")
        self.assertFalse(result["output_safe"])


if __name__ == "__main__":
    unittest.main()
