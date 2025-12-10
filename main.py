import argparse
import logging
import sys
from pathlib import Path

from document_processor import load_job_description
from llm_scorer import initialize_llm
from ranker import process_all_resumes, rank_and_save_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score and rank resumes against a job description using a local LLM."
    )
    parser.add_argument(
        "--data",
        default="./data/",
        help="Path to folder containing resume files (default: ./data/).",
    )
    parser.add_argument(
        "--jd",
        default="./jd.txt",
        help="Path to the job description file (default: ./jd.txt).",
    )
    parser.add_argument(
        "--output",
        default="./results.txt",
        help="Path to write ranking results (default: ./results.txt).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()
    data_folder = Path(args.data)
    jd_path = Path(args.jd)
    output_path = Path(args.output)

    if not data_folder.exists() or not data_folder.is_dir():
        print(f"Data folder not found or not a directory: {data_folder}")
        sys.exit(1)

    try:
        print(f"Loading job description from {jd_path}...")
        job_description = load_job_description(str(jd_path))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading job description: {exc}")
        sys.exit(1)

    print("Initializing LLM (qwen3:8b-q4_K_M)...")
    try:
        llm = initialize_llm()
    except Exception as exc:
        print(f"Failed to initialize LLM: {exc}")
        sys.exit(1)

    print(f"Scanning {data_folder} for resumes...")
    results = process_all_resumes(str(data_folder), job_description, llm)
    if not results:
        print("No resumes were successfully processed.")
        sys.exit(1)

    print("Ranking results...")
    rank_and_save_results(results, str(output_path), job_description)

    print(f"Complete! {len(results)} resumes processed and ranked.")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
