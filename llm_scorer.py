from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are an expert technical recruiter evaluating resumes against the job description.

First read the job description and identify the must-have skills, tools, seniority, and domain context. Then assess how well the resume proves each item with concrete evidence (projects, accomplishments, impact).

Scoring rubric (0-100, be strict):
- Core skills/tools/tech stack match: 50
- Relevant experience level, scope, and years: 30
- Domain/industry alignment and impact: 10
- Education/certifications fit: 10

Job Description:
{job_description}

Resume:
{resume_text}

Return only JSON in this exact format:
{{
  "score": <number between 0-100>,
  "reasoning": "Concise justification citing the strongest matches and the top gaps."
}}"""


def initialize_llm() -> ChatOllama:
    """Initialize ChatOllama with the qwen3:8b-q4_K_M model."""
    return ChatOllama(model="qwen3:8b-q4_K_M", temperature=0.1)


def _clamp_score(value: float) -> float:
    return max(0.0, min(100.0, value))


def _parse_response_text(raw_text: Any) -> Dict[str, Any]:
    text = raw_text if isinstance(raw_text, str) else str(raw_text)
    extracted_text = text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        extracted_text = match.group(0)

    score: float = 0.0
    reasoning: str = text.strip()

    try:
        payload = json.loads(extracted_text)
        score = float(payload.get("score", 0.0))
        reasoning = str(payload.get("reasoning", reasoning)).strip() or reasoning
    except Exception:
        number_match = re.search(r"(\d+(\.\d+)?)", text)
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
