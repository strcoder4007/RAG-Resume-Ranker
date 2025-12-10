from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "You are an expert resume evaluator. Score this resume against the job description.\n\n"
    "Job Description:\n{job_description}\n\n"
    "Resume:\n{resume_text}\n\n"
    "Provide your evaluation in the following JSON format:\n"
    '{\n  "score": <number between 0-100>,\n  "reasoning": "<brief explanation>"\n}\n'
    "Focus on: skills match, experience relevance, qualifications alignment."
)


def initialize_llm() -> ChatOllama:
    """Initialize ChatOllama with the qwen3:8b-q4_K_M model."""
    return ChatOllama(model="qwen3:8b-q4_K_M", temperature=0.1)


def _clamp_score(value: float) -> float:
    return max(0.0, min(100.0, value))


def _parse_response_text(raw_text: str) -> Dict[str, Any]:
    extracted_text = raw_text
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        extracted_text = match.group(0)

    score: float = 0.0
    reasoning: str = raw_text.strip()

    try:
        payload = json.loads(extracted_text)
        score = float(payload.get("score", 0.0))
        reasoning = str(payload.get("reasoning", reasoning)).strip() or reasoning
    except Exception:
        number_match = re.search(r"(\d+(\.\d+)?)", raw_text)
        if number_match:
            score = float(number_match.group(1))
        logger.warning("LLM response did not return valid JSON; using fallback parsing.")

    return {"score": _clamp_score(score), "reasoning": reasoning or "No reasoning provided."}


def score_resume(llm: ChatOllama, resume_text: str, job_description: str) -> Dict[str, Any]:
    """
    Score a single resume against the job description.

    Returns:
        A dictionary with numeric score and reasoning text.
    """
    if not resume_text.strip():
        return {"score": 0.0, "reasoning": "Resume text is empty after extraction."}

    prompt = PROMPT_TEMPLATE.format(
        job_description=job_description, resume_text=resume_text
    )

    try:
        response = llm.invoke(prompt)
    except Exception as exc:
        logger.error("LLM scoring failed: %s", exc)
        return {"score": 0.0, "reasoning": f"LLM scoring failed: {exc}"}

    raw_text = getattr(response, "content", str(response))
    return _parse_response_text(raw_text)
