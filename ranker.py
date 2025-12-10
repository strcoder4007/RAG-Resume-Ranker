from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from document_processor import process_document
from llm_scorer import score_resume

logger = logging.getLogger(__name__)


def _gather_resume_files(data_folder: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in data_folder.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pdf", ".docx"}
        ]
    )


def process_all_resumes(
    data_folder: str,
    job_description: str,
    llm: Any,
    output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Process all resumes in the data folder and score them.

    Returns:
        A list of dictionaries containing filename, score, reasoning, and file_type.
    """
    folder_path = Path(data_folder)
    resume_files = _gather_resume_files(folder_path)
    total = len(resume_files)
    results: List[Dict[str, Any]] = []

    if total == 0:
        logger.warning("No resume files found in %s", data_folder)
        return results

    pdf_count = sum(1 for path in resume_files if path.suffix.lower() == ".pdf")
    docx_count = sum(1 for path in resume_files if path.suffix.lower() == ".docx")
    print(f"Found {total} documents ({pdf_count} PDF, {docx_count} DOCX)")
    with tqdm(resume_files, desc="Scoring resumes", unit="resume") as progress:
        for path in progress:
            progress.set_postfix(file=path.name)

            doc_data = process_document(str(path))
            if not doc_data:
                progress.write(f"Skipped {path.name}: unreadable or empty.")
                continue

            try:
                scoring = score_resume(llm, doc_data["content"], job_description)
            except Exception as exc:
                logger.error("LLM scoring failed for %s: %s", path.name, exc)
                progress.write(f"Failed to score {path.name}: {exc}")
                continue

            scoring.update(
                {"filename": doc_data["filename"], "file_type": doc_data["file_type"]}
            )
            results.append(scoring)
            progress.write(f"Scored {path.name}: {scoring['score']:.1f}")

            if output_path:
                rank_and_save_results(results, output_path, job_description)

    return results


def rank_and_save_results(
    results: List[Dict[str, Any]], output_path: str = "./results.txt", job_description: str = ""
) -> None:
    """Sort results by score and write them to a results file."""
    sorted_results = sorted(results, key=lambda item: item.get("score", 0.0), reverse=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    jd_preview = job_description[:100]
    if len(job_description) > 100:
        jd_preview += "..."

    lines: List[str] = [
        "===========================================",
        "RESUME RANKING RESULTS",
        "===========================================",
        f"Total Resumes Processed: {len(results)}",
        f"Job Description: {jd_preview}",
        f"Generated: {timestamp}",
        "",
        "-------------------------------------------",
        "RANKED RESULTS",
        "-------------------------------------------",
        "",
    ]

    for index, record in enumerate(sorted_results, start=1):
        lines.append(f"Rank {index}: {record.get('filename', 'unknown')}")
        lines.append(f"Score: {record.get('score', 0.0):.1f}/100")
        reasoning = record.get("reasoning", "No reasoning provided.").strip()
        lines.append(f"Reasoning: {reasoning}")
        lines.append("")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    logger.info("Results written to %s", output_path)
