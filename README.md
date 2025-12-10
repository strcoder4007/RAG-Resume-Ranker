# Resume Ranker CLI

Command-line tool to score and rank resumes (PDF/DOCX) against a job description using a local Ollama model.

## Requirements
- Python 3.8+
- Ollama running locally with the `qwen3:8b-q4_K_M` model
- Dependencies from `requirements.txt`

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

ollama serve
ollama pull qwen3:8b-q4_K_M
```

## Usage
Place resume files in `./data/` and add the job description to `./jd.txt` (defaults). Then run:
```bash
python main.py
```

Optional flags:
```bash
python main.py --data ./data/ --jd ./jd.txt --output ./results.txt
```

## Output
- Ranked results are written to `./results.txt`, sorted by score (highest first) with brief reasoning.
- Progress is printed to the console while processing each resume.

## Notes
- Corrupted or unreadable files are skipped with a warning.
- If the LLM response is not valid JSON, a fallback parser attempts to extract a score and uses the raw response as reasoning.
