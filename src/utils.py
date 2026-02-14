import json
import re
from typing import Any, Dict, Type, TypeVar

_JSON_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)

T = TypeVar("T")


def extract_json(text: str) -> str:
    """
    Best-effort: extract first JSON object/array from a string.
    """
    if text is None:
        raise ValueError("Model output is None. Your LLM.complete() must return a JSON string.")
    m = _JSON_RE.search(text.strip())
    if not m:
        raise ValueError(f"No JSON found in model output. Output was:\n{text}")
    return m.group(1)


def loads_json(text: str) -> Dict[str, Any] | Any:
    s = extract_json(text)
    return json.loads(s)


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def complete_and_validate(
    llm: Any,
    system: str,
    user_json: str,
    model_cls: Type[T],
    retries: int = 2
) -> T:
    """
    Preferred path:
      - If llm implements complete_structured(system, user, model_cls) -> model instance,
        use Structured Outputs (schema-guaranteed) and return the parsed model.
    Fallback path:
      - Otherwise use llm.complete(...) -> JSON string + parse + Pydantic model_validate.

    No rule-based fallback in this function.
    """
    # 1) Structured Outputs path (recommended for OpenAI)
    if hasattr(llm, "complete_structured"):
        # Expect llm.complete_structured to return a Pydantic model instance
        return llm.complete_structured(system=system, user=user_json, model_cls=model_cls)

    # 2) JSON-mode path (valid JSON but not schema-guaranteed) + retries
    last_err: Exception | None = None
    cur_user = user_json

    for _ in range(retries + 1):
        raw = llm.complete(system=system, user=cur_user)

        if raw is None:
            last_err = ValueError("LLM returned None. llm.complete() must return a JSON string.")
        else:
            try:
                data = loads_json(raw)
                return model_cls.model_validate(data)
            except Exception as e:
                last_err = e

        cur_user = (
            user_json
            + "\n\n"
            + f"IMPORTANT: Your previous output was invalid. Error: {last_err}. "
              "Return JSON ONLY that matches the required schema. No markdown, no commentary."
        )

    raise last_err if last_err else RuntimeError("Unknown LLM validation error")
