import logging
from typing import Iterable, Literal


GuardrailStatus = Literal["PASSED", "BLOCKED", "SKIPPED", "ERROR"]

_terminal_logger = logging.getLogger("uvicorn.error")
_uvicorn_logger = logging.getLogger("uvicorn")


def _write_terminal_message(message: str) -> None:
    """Write through Uvicorn when available, otherwise print directly."""
    if _uvicorn_logger.handlers:
        _terminal_logger.info(message)
    else:
        print(message, flush=True)


def report_guardrail_result(
    agent_name: str,
    guardrail_name: str,
    status: GuardrailStatus,
) -> None:
    """Write a content-free guardrail result to the active service terminal."""
    _write_terminal_message(
        f"[Guardrail][{agent_name}] {guardrail_name}: {status}"
    )


def report_tool_usage(agent_name: str, tool_names: Iterable[str]) -> None:
    """Write the tools selected for one prompt to the service terminal."""
    invoked_tools = list(tool_names)
    selection = ", ".join(invoked_tools) if invoked_tools else "none"
    _write_terminal_message(
        f"[Tools][{agent_name}] Invoked: {selection}"
    )
