from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Type

from pydantic import BaseModel

try:
    from src.utils import get_logger as _get_logger  
except ImportError:  
    _get_logger = None

try:
    from src.config import get_llm_config
except ImportError as exc:  
    raise RuntimeError("Configuration module is required for LLM orchestration") from exc


logger = _get_logger(__name__) if callable(_get_logger) else logging.getLogger("LLM")

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that must rely solely on the provided context snippets.\n"
    "When the question is simple, respond as concisely as possible.\n"
    "If the question requires explanation, build the reasoning strictly from the supplied chunks.\n"
    "If the answer is missing from the context, reply with: \"No data in the documents.\""
)

PROMPT_TEMPLATE = (
    "You answer based only on the context below.\n"
    "If the answer is not present in the context,\n"
    "respond: \"No data in the documents.\"\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{query}"
)


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return ""
    formatted_parts = [f"[{idx + 1}] {chunk.get('text', '').strip()}" for idx, chunk in enumerate(chunks)]
    return "\n\n".join(formatted_parts)


def _call_openai(
    prompt: str,
    system_prompt: str,
    query: str,
    model: str,
    temperature: float,
    api_key: str,
    response_model: Type[BaseModel] | None = None,
) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:  
        raise RuntimeError("openai package is required for OpenAI provider") from exc

    base_client = OpenAI(api_key=api_key)
    if response_model is not None:
        try:
            import instructor
        except ImportError as exc:
            raise RuntimeError("instructor package is required for structured OpenAI responses") from exc

        client = instructor.patch(base_client, mode=instructor.Mode.MD_JSON)
    else:
        client = base_client

    try:
        start = time.perf_counter()
        logger.info("LLM | Sending request to OpenAI question: %s...", query)
        if response_model is not None:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                temperature=temperature,
                response_format=response_model,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info("LLM | OpenAI request completed in %.2f ms", elapsed_ms)
            parsed = response.choices[0].message.parsed
            logger.debug("LLM | OpenAI parsed response: %s", parsed)
            return parsed
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
            temperature=temperature,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("LLM | OpenAI request completed in %.2f ms", elapsed_ms)
        return response.choices[0].message.content or ""
    except Exception as exc:  
        raise RuntimeError(f"OpenAI completion failed: {exc}") from exc


def _call_gemini(prompt: str, system_prompt: str, query: str, model: str, temperature: float, api_key: str) -> str:
    try:
        import google.generativeai as genai
    except ImportError as exc:  
        raise RuntimeError("google-generativeai package is required for Gemini provider") from exc

    genai.configure(api_key=api_key)
    try:
        generative_model = genai.GenerativeModel(model)
        full_prompt = f"{system_prompt}\n\n{prompt}"
        start = time.perf_counter()
        response = generative_model.generate_content(full_prompt, generation_config={"temperature": temperature})
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("LLM | Gemini request completed in %.2f ms", elapsed_ms)
    except Exception as exc:  
        raise RuntimeError(f"Gemini completion failed: {exc}") from exc

    return getattr(response, "text", "")


def _estimate_tokens(payload: Any) -> int:
    if isinstance(payload, str):
        return len(payload.split())
    if isinstance(payload, BaseModel):
        return len(payload.model_dump_json().split())
    return len(str(payload).split())


def _invoke_llm(
    query: str,
    context_chunks: List[Dict[str, Any]],
    provider: str | None,
    model: str | None,
    temperature: float,
    system_prompt: str | None,
    response_model: Type[BaseModel] | None,
) -> Any:
    config = get_llm_config(provider)
    provider_key = config.provider
    context = _format_context(context_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)
    final_system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    chosen_model = model or config.model

    context_len = len(context)
    prompt_len = len(prompt)
    prompt_tokens = len(prompt.split())

    logger.info(
        "LLM | provider=%s model=%s context_chunks=%d context_chars=%d",
        provider_key,
        chosen_model,
        len(context_chunks),
        context_len,
    )
    logger.info("LLM | prompt_len=%d approx_tokens=%d", prompt_len, prompt_tokens)
    logger.debug("LLM | prompt preview: %s", prompt[:300])

    if provider_key == "openai":
        response = _call_openai(
            prompt,
            final_system_prompt,
            query,
            chosen_model,
            temperature,
            config.api_key,
            response_model=response_model,
        )
    elif provider_key == "gemini":
        if response_model is not None:
            raise ValueError("Structured outputs are not supported for Gemini provider.")
        response = _call_gemini(prompt, final_system_prompt, query, chosen_model, temperature, config.api_key)
    else:  
        raise ValueError(f"Unsupported provider: {provider}")

    tokens = _estimate_tokens(response)
    preview: str
    if isinstance(response, BaseModel):
        preview = getattr(response, "content", response.model_dump_json())[:200]
    else:
        preview = str(response)[:200]
    logger.llm("LLM | response preview: %s", preview)
    logger.info("LLM | answer size approx_tokens=%d", tokens)
    return response


def generate_answer(
    query: str,
    context_chunks: List[Dict[str, Any]],
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 1.0,
    system_prompt: str | None = None,
) -> str:
    response = _invoke_llm(
        query=query,
        context_chunks=context_chunks,
        provider=provider,
        model=model,
        temperature=temperature,
        system_prompt=system_prompt,
        response_model=None,
    )
    if isinstance(response, str):
        return response
    if isinstance(response, BaseModel):
        return getattr(response, "content", response.model_dump_json())
    return str(response)


def generate_structured_answer(
    query: str,
    context_chunks: List[Dict[str, Any]],
    response_model: Type[BaseModel],
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 1.0,
    system_prompt: str | None = None,
) -> BaseModel:
    response = _invoke_llm(
        query=query,
        context_chunks=context_chunks,
        provider=provider,
        model=model,
        temperature=temperature,
        system_prompt=system_prompt,
        response_model=response_model,
    )
    if not isinstance(response, BaseModel):
        raise RuntimeError("Structured LLM call did not return a BaseModel instance.")
    return response


if __name__ == "__main__":  
    demo_chunks = [
        {"chunk_id": "1", "text": "Python is often used for rapid prototyping."},
        {"chunk_id": "2", "text": "NumPy arrays enable fast numerical operations."},
    ]
    result = generate_answer("What is Python used for?", demo_chunks, provider="openai")
    print(result)
