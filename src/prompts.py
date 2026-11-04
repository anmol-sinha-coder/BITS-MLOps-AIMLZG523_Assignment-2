"""The three prompt versions evaluated by the pipeline (Task 3 of the assignment).

Each version is registered in the MLflow Prompt Registry with a meaningful
commit message, run against the same five questions, and compared on latency,
tokens, cost and JSON-validity before the best one is promoted to the
`production` alias.
"""
from __future__ import annotations

# Jinja-style placeholder kept identical to the assignment specification so the
# diff between versions is visible in the MLflow Prompt Registry.
QUESTION_PLACEHOLDER = "{{question}}"

PROMPTS: dict[str, dict] = {
    "v1": {
        "name": "eduassist_basic",
        "commit": "Basic prompt: open-ended answer with no persona or format constraints",
        "template": (
            "Answer the following student question clearly.\n"
            f"Question: {QUESTION_PLACEHOLDER}"
        ),
    },
    "v2": {
        "name": "eduassist_structured",
        "commit": "Structured persona prompt for concise undergraduate-level prose",
        "template": (
            "You are an academic assistant.\n"
            "Answer the following student question in a clear, concise, and accurate manner.\n"
            "Use simple language suitable for undergraduate students.\n"
            f"Question: {QUESTION_PLACEHOLDER}"
        ),
    },
    "v3": {
        "name": "eduassist_json_output",
        "commit": "Strict JSON-output prompt with fixed schema, validated by valid_json",
        "template": (
            "You are an academic assistant.\n"
            "Answer the student question and return ONLY valid JSON.\n"
            "Do not include markdown, explanations outside JSON, or code fences.\n"
            "Use this exact JSON schema:\n"
            "{\n"
            '  "answer": "string",\n'
            '  "confidence": "high | medium | low",\n'
            '  "key_points": ["string"]\n'
            "}\n"
            f"Question: {QUESTION_PLACEHOLDER}"
        ),
    },
}


def render_prompt(template: str, question: str) -> str:
    """Substitute the question placeholder into a prompt template."""
    return template.replace(QUESTION_PLACEHOLDER, question)


def is_json_version(version_key: str) -> bool:
    """Only v3 is expected to return strict JSON."""
    return version_key == "v3"
