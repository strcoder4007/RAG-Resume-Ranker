# Handoff Document — RAG-Resume-Ranker

## Project Overview

**What it does:** A command-line tool that scores and ranks hundreds of resumes (PDF/DOCX) against a job description using a local Ollama LLM (`qwen3:8b-q4_K_M`). It processes all resumes in a data folder, calls the LLM with a structured scoring prompt, extracts a numeric score (0–100) and reasoning, sorts by score, and writes results to a file.

**Target hardware:** Any machine running Ollama with `qwen3:8b-q4_K_M` installed. No GPU required (CPU inference works fine for 8B quantized models).

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Ollama — `ChatOllama` (LangChain) |
| Document parsing | `PyPDF2` (PDF), `python-docx` (DOCX) |
| CLI | Built-in `argparse` |
| Configuration | `.env` (optional, via python-dotenv) |
| Output | Plain text file (`./results.txt`) |
| Logging | Python built-in `logging` |

---

## Project Structure

```
RAG-Resume-Ranker/
├── README.md                  # Usage documentation
├── requirements.txt           # Dependencies: langchain-ollama, PyPDF2, python-docx, tqdm
├── main.py                    # CLI entry point: argument parsing, job description loading, LLM init, orchestrates ranking
├── document_processor.py      # PDF/DOCX text extraction + job description loading + process_document() function
├── ranker.py                  # Core ranking logic: file discovery, parallel processing via tqdm, scoring, result saving
├── llm_scorer.py              # LLM initialization, scoring prompt, response parsing + fallback JSON extraction
└── data/                      # Directory containing resume PDF/DOCX files
    └── (resume files)
├── jd.txt                     # Job description file (default)
└── results.txt                # Output ranking file (default)
```

---

## How It Works — Detailed Architecture

### Pipeline Flow

```
main.py (parse args)
    ↓
load_job_description(jd.txt)          ← document_processor.py
    ↓
initialize_llm()                      ← llm_scorer.py (qwen3:8b-q4_K_M, temp=0.1)
    ↓
process_all_resumes()                 ← ranker.py
    ├── _gather_resume_files()        ← glob for .pdf/.docx in data folder
    └── tqdm loop over files:
            process_document()        ← document_processor.py → {filename, content, file_type}
            score_resume()            ← llm_scorer.py → {score, reasoning}
            results.append()
    ↓
rank_and_save_results()               ← ranker.py
    ├── sort by score descending
    └── write to results.txt
```

### Scoring Prompt Design

The prompt in `llm_scorer.py` instructs the LLM to:
1. Read the job description and identify must-have skills, tools, seniority, and domain context
2. Assess how well the resume proves each item with concrete evidence
3. Apply a strict rubric:
   - Core skills/tools/tech stack match: **50 points**
   - Relevant experience level, scope, and years: **30 points**
   - Domain/industry alignment and impact: **10 points**
   - Education/certifications fit: **10 points**
4. Return JSON: `{ "score": <0-100>, "reasoning": "<justification>" }`

### Response Parsing

The `_parse_response_text()` function in `llm_scorer.py` handles LLM outputs that may not be clean JSON:
1. First attempts `json.loads()` on the full response
2. Falls back to extracting the first `{...}` block via regex
3. If JSON parsing fails entirely, uses regex to extract the first number as a score and the raw text as reasoning
4. Clamps the score to [0.0, 100.0] via `_clamp_score()`

### Parallel Processing

Resumes are processed sequentially (not in parallel) via `tqdm`. Each resume:
1. Is loaded and text-extracted
2. Passed to `score_resume()` which calls the LLM
3. The LLM call is the bottleneck; sequential processing avoids overwhelming the Ollama server

### Result Output Format

`results.txt` format:
```
===========================================
RESUME RANKING RESULTS
===========================================
Total Resumes Processed: N
Job Description: <first 100 chars>...
Generated: <timestamp>

-------------------------------------------
RANKED RESULTS
-------------------------------------------

Rank 1: resume.pdf
Score: 85.0/100
Reasoning: Strong Python and ML experience...

Rank 2: resume2.docx
Score: 72.5/100
...
```

---

## Current State / What's Working

- **End-to-end pipeline functional**: Drop resumes in `./data/`, write JD in `./jd.txt`, run `python main.py`, get ranked results.
- **PDF and DOCX support**: Both formats are extracted correctly.
- **Fallback parsing**: Robust against malformed LLM JSON responses.
- **Progress display**: tqdm shows per-file progress with score updates.
- **Ink results written incrementally**: `rank_and_save_results()` is called after every resume (via the `output_path` parameter in `process_all_resumes()`), so partial results are available even if the run is interrupted.

### Known Gaps / Rough Edges

1. **No vector store / semantic search**: Not a RAG system in the traditional sense — no embeddings, no Chroma, no retrieval. The "RAG" in the name is aspirational; it's really just LLM-based scoring.
2. **No batching of LLM calls**: Each resume is a separate LLM call. For 100+ resumes, this is slow (sequential).
3. **No caching**: If the same resume is scored twice with the same JD, it still calls the LLM both times.
4. **No resume metadata**: The scoring prompt has no information about the candidate (name, email, years of experience) — only the text content.
5. **No confidence calibration**: All resumes get scores but there's no measure of how certain the model is.
6. **No interview shortlist cut-off**: All resumes are ranked but no recommendation on where to cut off.
7. **No multi-JD comparison**: Can't compare the same resume against multiple job descriptions in one run.
8. **No structured output schema**: The job description is just raw text; no parsing of required skills into structured fields.

---

## Improvements

### 1. Add Vector Store and Semantic Search (True RAG)
**What to do:** Add ChromaDB-backed semantic search to actually make this a RAG system. When the job description is loaded, embed it and store it in a collection. Then, when scoring each resume, retrieve the most relevant JD sections for the candidate's background and include those as context in the scoring prompt.

Implement in `document_processor.py` a new `embed_jd()` function that embeds the job description and stores it in Chroma. Update `score_resume()` in `llm_scorer.py` to first retrieve relevant JD chunks for each resume's content, then include those in the scoring prompt.

Add config in `ranker.py` for the vector store path and retrieval top-k.

**Why it matters:** The current system treats the JD as flat text. Retrieval-augmented scoring would match resume sections to the most relevant JD requirements, producing more accurate scores. This makes the "RAG" in the name accurate. Staff Engineer context: RAG architecture, vector store integration, retrieval-augmented scoring pipelines.

**Files likely to touch:** `document_processor.py` (add embedding logic), `llm_scorer.py` (update scoring prompt to include retrieved context), new `vector_store.py` (optional module), `requirements.txt` (add `langchain-chroma`).

**Verification:** Score the same resume with and without retrieval augmentation; compare whether the retrieval-augmented score is more accurate against a manual assessment.

---

### 2. Add Batch Processing for LLM Calls
**What to do:** Instead of calling the LLM once per resume sequentially, batch multiple resumes into a single LLM call using a multi-shot prompt. For example, send 5 resumes at once with a prompt that asks the LLM to score all 5 and return a JSON array.

Implement a `score_batch()` function in `llm_scorer.py` that accepts a list of `(filename, resume_text)` tuples and returns a list of `(score, reasoning)` results. Use a prompt that structures the input as a numbered list of resumes and asks for a JSON object with a `results` array.

Add a `--batch-size` CLI flag in `main.py` (default 3–5 depending on context length).

**Why it matters:** For 100 resumes, sequential processing means 100 round-trips to the LLM. Batching reduces this to 20–33 round-trips, dramatically improving throughput. Staff Engineer context: batch inference, throughput optimization, context management in LLM pipelines.

**Files likely to touch:** `llm_scorer.py` (new `score_batch()` function), `ranker.py` (batch loop instead of single-item loop), `main.py` (add `--batch-size` flag).

**Verification:** Run with `--batch-size 5` and measure time per resume; compare against sequential processing.

---

### 3. Add Interview Shortlist Recommendation
**What to do:** After ranking all resumes, add an analysis step that recommends a shortlist cut-off based on score distribution. Compute the score distribution (mean, median, std dev) and flag a natural break point (e.g., "Top 10 resumes score ≥ 70, clearly above the next tier").

Add this analysis to `rank_and_save_results()` in `ranker.py`, producing a shortlist recommendation section in `results.txt`:

```
-------------------------------------------
SHORTLIST RECOMMENDATION
-------------------------------------------
Top 12 resumes (score ≥ 75) are recommended for interview.
Next 20 resumes (score 60–75) are borderline — review if time permits.
```

Make the cut-off threshold configurable via `--shortlist-threshold` CLI flag.

**Why it matters:** HR teams want a clear shortlist, not just a sorted list. Automated shortlisting reduces manual review effort. Staff Engineer context: statistical analysis of score distributions, threshold-based decision systems, UX-oriented output design.

**Files likely to touch:** `ranker.py` (`rank_and_save_results` — add shortlist analysis).

**Verification:** Run on a dataset of 50+ resumes; confirm the shortlist recommendation reflects a genuine score gap.

---

### 4. Add Per-Skill Scoring Breakdown
**What to do:** Extend the scoring prompt to return not just a single score but a per-skill breakdown: `{ "overall_score": 82, "skill_scores": {"Python": 9, "LLM": 8, "System Design": 6, "Communication": 7}, "reasoning": "..." }`.

Update `score_resume()` to parse this extended format. Update `rank_and_save_results()` to display the per-skill breakdown and sort by a weighted combination (configurable skill weights) rather than a single overall score.

Add a `--skill-weights` CLI flag in JSON format: `'{"Python": 2, "LLM": 3}'` that multiplies skill scores before aggregating.

**Why it matters:** A single overall score hides important information. Knowing that a candidate is strong in Python but weak in system design is more actionable for hiring. Staff Engineer context: multi-dimensional evaluation, structured output parsing, weighted scoring systems.

**Files likely to touch:** `llm_scorer.py` (update prompt and parsing), `ranker.py` (update sorting and display), `main.py` (add `--skill-weights` flag).

**Verification:** Score a resume and confirm the per-skill breakdown appears in the output.

---

### 5. Add Resume Deduplication by Content Similarity
**What to do:** Before scoring, compute embeddings of all resume texts and detect near-duplicates using cosine similarity. If two resumes score ≥ 0.95 similarity, keep the one with the higher score and flag the other as a duplicate.

Add this as a pre-processing step in `process_all_resumes()` before the scoring loop. Use the Ollama embedder (same model used in lite-deep-research-agent) to embed each resume's text. Implement deduplication in a new `dedupe_resumes()` function in `ranker.py`.

Print a warning when duplicates are found: `"Skipping duplicate: resume2.pdf (96% similar to resume1.pdf)"`.

**Why it matters:** Candidates sometimes submit multiple versions of their resume. Scoring duplicates separately skews results and wastes LLM calls. Staff Engineer context: deduplication algorithms, embedding-based similarity, data cleaning pipelines.

**Files likely to touch:** `ranker.py` (add deduplication logic), `requirements.txt` (verify langchain-ollama is available for embeddings).

**Verification:** Place two near-identical PDFs in `data/`; confirm only one is scored and the other is flagged as duplicate.

---

### 6. Add Structured JD Parsing
**What to do:** Instead of passing raw JD text to the LLM, parse it into structured fields: required skills (list), preferred skills (list), years of experience, education requirements, domain knowledge. Then score against these structured criteria rather than the full raw text.

Implement `parse_job_description()` in `document_processor.py` that uses the LLM (or a lightweight parse) to extract structured fields from the JD text. Update the scoring prompt in `llm_scorer.py` to explicitly reference the parsed criteria.

This enables per-skill scoring and better calibration because the criteria are explicit rather than embedded in long prose.

**Why it matters:** Raw JD text introduces noise — the LLM must extract requirements from prose. Structured parsing makes requirements explicit and enables more precise scoring. Staff Engineer context: structured information extraction, LLM-based parsing, schema validation.

**Files likely to touch:** `document_processor.py` (add `parse_job_description()`), `llm_scorer.py` (update scoring prompt), possibly `requirements.txt` (add `pydantic` for structured output).

**Verification:** Run and confirm the JD is parsed into explicit skill lists; compare scoring consistency before and after.

---

### 7. Add Score Confidence and Calibration
**What to do:** Add a second LLM call after scoring that asks: "How confident are you in this score? Rate your confidence low/medium/high and explain why." Return this as a `confidence` field alongside the score.

In `rank_and_save_results()`, flag low-confidence scores with a `⚠️` marker. This helps HR focus on high-confidence recommendations and investigate low-confidence ones.

Implement this in `llm_scorer.py` as a two-pass approach: first score, then calibrate. Or use a single prompt that returns `{score, reasoning, confidence}`.

**Why it matters:** Not all scores are equally reliable. A score of 72 from a resume with unclear experience descriptions should be treated differently than a 72 from a well-documented resume. Staff Engineer context: uncertainty quantification, calibration, trust in ML outputs.

**Files likely to touch:** `llm_scorer.py` (update prompt and parsing), `ranker.py` (display confidence markers).

**Verification:** Score a vague resume vs. a detailed one; confirm the vague one gets a low-confidence flag.

---

### 8. Add Results Summary Statistics
**What to do:** At the end of `rank_and_save_results()`, add a statistics section:

```
-------------------------------------------
SCORE STATISTICS
-------------------------------------------
Total resumes: 127
Mean score: 58.3
Median score: 61.0
Std deviation: 12.4
Top score: 89.0 (resume42.pdf)
Bottom score: 23.0 (resume7.pdf)
Score distribution:
  80-100: 8 resumes
  60-80:  42 resumes
  40-60:  55 resumes
  20-40:  18 resumes
  0-20:   4 resumes
```

Use Python's `statistics` module for mean/median/std. Use a simple histogram with `collections.Counter` binning.

**Why it matters:** Summary statistics help HR understand the overall candidate pool quality and whether the bar needs adjustment. Staff Engineer context: statistical analysis, data summarization, reporting.

**Files likely to touch:** `ranker.py` (`rank_and_save_results` — add statistics section).

**Verification:** Run and confirm the statistics section appears at the end of `results.txt`.

---

### 9. Add Caching to Avoid Rescoring Unchanged Resumes
**What to do:** Implement a hash-based cache: compute `SHA256(resume_text + jd_text)` and store the score in `~/.resume-ranker/cache.json`. Before scoring, check if an identical hash exists and return the cached score.

Add a `--no-cache` flag to bypass the cache, and a `--clear-cache` flag to wipe it. Implement the cache in a new `cache.py` module.

**Why it matters:** Iterative use of the tool (refining JD, re-running) causes unnecessary rescoring of unchanged resumes. Caching eliminates this waste. Staff Engineer context: caching strategies, hash-based deduplication, state management.

**Files likely to touch:** New `cache.py`, `ranker.py` (check cache before scoring), `main.py` (add cache-related flags).

**Verification:** Run twice on the same dataset; confirm the second run is significantly faster and prints "Cache hit for X/Y resumes".

---

### 10. Add Web UI (Gradio) for Non-Technical Users
**What to do:** Create a Gradio web interface (`webui.py`) with:
- File upload area for the data folder (or drag-and-drop ZIP of resumes)
- Text area for job description (with file loading option)
- "Run Ranking" button with progress display
- Results table with sortable columns (rank, filename, score, reasoning)
- Download button for results.txt

This mirrors the pattern in `browser-agent` project's web UI. Add a `run_webui()` entry point.

**Why it matters:** Non-technical HR users cannot use the CLI. A web UI dramatically expands adoption. Staff Engineer context: web UI development, Gradio integration, accessibility-oriented design.

**Files likely to touch:** New `webui.py`, updated `main.py` (add `run_webui()` entry point), `requirements.txt` (add `gradio`).

**Verification:** Launch the web UI; upload a ZIP of resumes; run ranking and confirm results appear in the table.