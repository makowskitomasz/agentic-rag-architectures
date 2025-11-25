from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import instructor
from pydantic import ValidationError

try:
    from src.utils import get_logger as _get_logger  
except ImportError:  
    _get_logger = None

try:
    from src.config import get_llm_config
except ImportError as exc:  
    raise RuntimeError("Configuration module is required for evaluation") from exc

logger = _get_logger(__name__) if callable(_get_logger) else logging.getLogger(__name__)

PROMPT_TEMPLATE = """
You are an evaluation assistant. Your task is to judge whether the given answer
is well-grounded in the provided context or if it likely contains hallucinations.

Given the context and the answer, evaluate the grounding and hallucination risk.

IMPORTANT:
- "grounded": true if the key claims in the answer are directly supported by the context.
- "hallucination": true if the answer introduces unsupported claims or contradicts the context.
- "confidence": integer from 1 (low confidence) to 5 (very high confidence).

Return ONLY valid JSON in the following format:
{{"grounded": <true/false>, "hallucination": <true/false>, "confidence": <1-5>}}

Context:
{context}

Answer:
{answer}
""".strip()



def _format_context(chunks: List[Dict[str, Any]]) -> str:
    return "\n".join(f"[{idx + 1}] {chunk.get('text', '').strip()}" for idx, chunk in enumerate(chunks))


def _call_openai(prompt: str, model: str, temperature: float, api_key: str) -> EvaluationResult:
    try:
        from openai import OpenAI
    except ImportError as exc:  
        raise RuntimeError("openai package is required for evaluation") from exc

    client = instructor.patch(OpenAI(api_key=api_key))
    try:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an evaluation assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            response_model=EvaluationResult,
        )
    except Exception as exc:  
        raise RuntimeError(f"OpenAI evaluation call failed: {exc}") from exc


def _call_gemini(prompt: str, model: str, temperature: float, api_key: str) -> str:
    try:
        import google.generativeai as genai
    except ImportError as exc:  
        raise RuntimeError("google-generativeai package is required for evaluation") from exc

    genai.configure(api_key=api_key)
    try:
        generative_model = genai.GenerativeModel(model)
        response = generative_model.generate_content(prompt, generation_config={"temperature": temperature})
    except Exception as exc:  
        raise RuntimeError(f"Gemini evaluation call failed: {exc}") from exc

    return getattr(response, "text", "")


def _parse_response(raw: str) -> EvaluationResult:
    fallback = EvaluationResult.fallback()
    if not raw:
        logger.warning("Empty evaluation response received.")
        return fallback
    try:
        return EvaluationResult.model_validate_json(raw)
    except ValidationError:
        try:
            parsed = json.loads(raw)
            return EvaluationResult.model_validate(parsed)
        except Exception as exc:
            logger.warning("Failed to parse evaluation response: %s", exc)
            return fallback


def evaluate_answer(
    answer: str,
    context_chunks: List[Dict[str, Any]],
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    context = _format_context(context_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, answer=answer)
    config = get_llm_config(provider)
    provider_key = config.provider

    chosen_model = model or config.model

    if provider_key == "openai":
        result = _call_openai(prompt, chosen_model, temperature, config.api_key)
    elif provider_key == "gemini":
        raw_response = _call_gemini(prompt, chosen_model, temperature, config.api_key)
        logger.debug("Evaluation raw response: %s", raw_response)
        result = _parse_response(raw_response)
    else:  
        raise ValueError(f"Unsupported provider: {provider_key}")

    logger.info("Evaluation provider=%s model=%s chunks=%d", provider_key, chosen_model, len(context_chunks))
    return result.model_dump()


if __name__ == "__main__":
    demo_chunks = [
        {"chunk_id": "1", "text": "The athlete maintained a 45° angle during acceleration.", "score": 0.9, "metadata": {}},
        {"chunk_id": "2", "text": "Cadence rose to 4.6 strides per second by 30 m.", "score": 0.85, "metadata": {}},
    ]
    demo_answer = "The athlete accelerated with a 45-degree lean and reached 4.6 strides per second."
    result = evaluate_answer(demo_answer, demo_chunks, provider="openai")
    print(result)
