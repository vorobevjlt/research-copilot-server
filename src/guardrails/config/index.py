import os

from guardrails import Guard
from guardrails_ai.detect_pii import DetectPII
from guardrails_ai.prompt_injection_detector import PromptInjectionDetector
from guardrails_ai.toxic_language import ToxicLanguage

os.environ.setdefault(
    "OPENAI_BASE_URL",
    os.getenv("OPENAI_API_BASE", "https://api.proxyapi.ru/openai/v1"),
)

PII_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "IP_ADDRESS",
]

output_guard = Guard(
    name="output-pii-safety",
    description="Blocks PII in generated responses.",
).use(
    DetectPII(
        pii_entities=PII_ENTITIES,
        on_fail="exception",
        use_local=True,
    )
)

input_toxic_guard = Guard(
    name="input-toxicity-safety",
    description="Blocks toxic language in user prompts.",
).use(
    ToxicLanguage(
        threshold=0.3,
        on_fail="exception",
        use_local=True,
    )
)


input_injection_guard = Guard(
    name="input-prompt-injection-safety",
    description="Blocks attempts to override or manipulate the agent.",
).use(
    PromptInjectionDetector(
        llm_callable="gpt-5.6-luna",
        threshold=0.8,
        on_fail="exception",
    )
)
