# BESS Acoustic Intelligence (BAI)

Phase 1 is a local, resumable collection tool for discovering publicly accessible technical documents about noise from utility-scale battery energy storage systems. It uses Tavily for web search, inspects public landing pages for PDF links, and stores document provenance and collection status in SQLite.

The project does not bypass paywalls, CAPTCHA, robots.txt restrictions, or other access controls. Search-result URLs are queued separately from downloads so they can be reviewed first. No RAG, LLM extraction, or dashboard is included.

## Requirements

- Windows 10/11
- Python 3.11 or newer
- A Tavily API key for API-backed discovery (manual URL and local-file imports do not need one)

## Windows setup

Open PowerShell in this project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
Copy-Item .env.example .env
```

Open `.env` and set `TAVILY_API_KEY` to your own key. Keep `.env` private; it is excluded by `.gitignore`.

Alternatively, install from `requirements.txt` with `python -m pip install -r requirements.txt` and run the CLI from this directory.

## Configuration

- Edit `config/search_queries.toml` to add or change search queries. The initial list includes seven general searches and region-focused searches for the United States, United Kingdom, Australia, Canada, and continental Europe.
- Add known publicly accessible URLs to `config/seed_urls.toml` using `[[seeds]]` entries with `url` and optional `title` fields. Seeds are processed without Tavily.
- Copy `.env.example` to `.env` and adjust optional limits. `BAI_MAX_API_REQUESTS_PER_RUN` limits search calls per invocation, including retries; every call is recorded in the database. `BAI_MAX_RESULTS` is the maximum result count per Tavily query (1-20), and `BAI_SEARCH_DEPTH` accepts Tavily's `basic`, `advanced`, `fast`, or `ultra-fast` values.
- Crawl depth is bounded to 0-5; the maximum pages inspected per result/seed and HTTP timeout, retries, and rate delay are configurable in `.env`.

By default, the SQLite database is `data/processed/bai.sqlite3`, downloads go to `data/raw`, and logs go to `logs/bai.log`. These locations are ignored by Git.

## CLI

Run commands from the project directory after setup:

```powershell
python -m bai --help
python -m bai discover --dry-run
python -m bai discover
python -m bai download --dry-run
python -m bai download
python -m bai import-url "https://public.example.org/noise-assessment.pdf" --title "Noise assessment"
python -m bai import-file "C:\Research\assessment.pdf"
python -m bai stats
```

`discover` searches the configured queries and seed URLs, records search API usage, and queues unique PDF URLs. `discover --dry-run` prints results but does not queue them; API usage is still recorded. Review queued URLs with `download --dry-run` before running `download`. Discovery can be tuned per invocation:

```powershell
python -m bai discover --max-results 20 --search-depth advanced --max-api-requests 5 --crawl-depth 2
```

`import-url` queues a URL for a later download; `import-file` validates and copies a local PDF into the collection unless its contents are already present. Failed downloads remain queued for later retry, and partial transfers are resumed when the server supports HTTP range requests.

## Testing

Tests use mocked provider responses and local fixtures; they do not call Tavily or require an API key:

```powershell
python -m pytest
```

The standard suite skips the regression test that reads private locally downloaded PDFs. To check the five manually inspected source tables without writing extracted content to the database:

```powershell
$env:BAI_RUN_PDF_INTEGRATION = "1"
python -m pytest tests/test_local_pdf_integration.py
Remove-Item Env:BAI_RUN_PDF_INTEGRATION
```

## Collection and provenance

The database stores each document's original URL, referring page when available, title, discovery and download timestamps, file size, SHA-256, local path, download status, and duplicate relationship. Search query/provider attribution is stored separately so multiple discovery paths are retained for a URL. Duplicate URLs are collapsed before download; identical PDF contents are detected by hash after transfer.

This is an initial discovery and collection tool, not a guarantee of search completeness or a substitute for checking each source's terms and applicability.

## Phase 2 — Stage A: deterministic document intelligence

Stage A uses PyMuPDF to extract page-numbered text and metadata, identify acoustic-signal pages, and detect candidate tables. PDF text and table extraction are deterministic; no LLM or OCR service is used. Parser version `pymupdf-2` preserves raw cell text and coordinates alongside reconstructed logical cells, row/column spans, multi-row header references, frequency-band mappings, measurement wording, and alignment/review status. Frequency bands are recognized only from header-scoped evidence; numeric measurement cells are not treated as frequencies based on value alone.

Pages with images and little selectable text, pages without extractable text, table-like acoustic passages with no detected table, ambiguous table alignment, unspecified measurement types, and parser errors are retained with warnings and review flags. These statuses mean the content could not be fully assessed by this parser; they do not mean the source document lacks the information. Evidence-based heuristic classification distinguishes BESS acoustic assessments, manufacturer-issued technical documentation, assessments citing manufacturer specifications, academic or industry research, cross-industry acoustic technology, general BESS engineering documentation, and unassessed/irrelevant material. Direct BESS relevance requires focused evidence rather than an isolated mention. Scores are heuristic ranking aids, not calibrated probabilities; source excerpts and review status are retained, and documents are never deleted based on classification.

The intelligence schema is additive and versioned; Stage A.1 extends extracted-table storage while preserving Phase 1 tables and records. A pre-migration SQLite backup is created beside the database as `bai.sqlite3-phase2-v1-backup`; both the database and backup are ignored by Git. Parsing is cached by document ID, recorded SHA-256, and parser version, so unchanged complete or partial parses are skipped. A local file whose hash differs from its Phase 1 record is not parsed and is flagged for review. Repeat parse/classify runs continue to report outstanding review items.

```powershell
python -m bai parse --dry-run
python -m bai parse
python -m bai classify --dry-run
python -m bai classify
python -m bai intelligence-stats
```

`parse` processes downloaded, imported, and duplicate-content PDFs with local paths. `classify` classifies available parsed text and keeps a separate evidence record for each scored category. `intelligence-stats` separates latest-version document/page/table outcomes from cumulative records retained across parser and classifier versions. It reports successful table extraction, missing-text pages, parse failures, ambiguous tables, unspecified measurement types, and outstanding review items. The first intelligence command may initialize the additive schema and create its ignored backup, including when invoked with `--dry-run`.

The equipment, acoustic-measurement, mitigation, and source-evidence tables provide provenance-linked storage foundations. Stage A leaves those engineering tables empty; it does not infer values, mix sound power with sound pressure, or calculate combined decibel levels. Stage B1 adds typed Pydantic schemas and an offline candidate-review workflow, described below. LLM-assisted extraction, OCR, and live provider integrations remain deferred.

## Phase 2B — Stage B1: evidence-first schemas and benchmark review

Stage B1 defines typed Pydantic schemas for project facts, equipment, acoustic measurements, mitigation, document provenance, parsed evidence, and manually approved gold records. Unknown values remain null or explicitly unknown. Sound power, sound pressure, SRI, and insertion loss are distinct; reported weighting, frequency bandwidth, measurement basis, facility/per-unit scope, operating conditions, uncertainty, and extraction confidence are represented separately. A value marked eligible for cross-project comparison must meet explicit completeness checks. Extraction confidence is not treated as correctness.

Candidate generation reads only the existing deterministic parsed tables and stored source evidence. It does not open a model connection, make API requests, or incur usage charges. Candidates are stored in the separate, Git-ignored `data/private/benchmarks.sqlite3` database and remain pending until a reviewer acts. The benchmark generator only accepts clear table alignments and unambiguous numeric cells, balances selections across source documents, and preserves each original candidate. The gold-standard set is a separate table containing only manually approved records.

Review a candidate and its source quote/cell coordinates before approving or correcting it:

```powershell
python -m bai benchmark generate --limit 50
python -m bai benchmark list --status pending
python -m bai benchmark show cand-0123456789abcdef
python -m bai benchmark review cand-0123456789abcdef --action approve --reviewer "Reviewer Name" --reason "Checked against the cited PDF page"
python -m bai benchmark stats
python -m bai benchmark export
```

For a correction, create a local JSON file containing the corrected `measurement` object and `human_review_evidence` list, then pass it with `--action correct --correction-file <path>`. Each evidence item records the document ID, page, table reference, exact quote, and whether it was matched against extracted page text or visually checked in the PDF. Numeric corrections must be supported by that human-reviewed quote; the original deterministic candidate and every correction event remain unchanged in the review history. A later approval promotes the reviewed version to the separate gold set. Rejections are also append-only. `benchmark list`, `show`, and `stats` continue to show existing review items on repeat runs.

Provider-processing eligibility is document-level and default-deny. Inspect or change the audited decision with:

```powershell
python -m bai provider-eligibility list
python -m bai provider-eligibility set 17 --allow --reviewer "Reviewer Name" --basis "Explicit authorization verified for this document"
python -m bai provider-eligibility set 17 --deny --reviewer "Reviewer Name" --basis "Authorization withdrawn"
```

The eligibility history is in the additive SQLite schema v3 migration. The existing database is backed up and integrity-checked before the migration; Phase 1 and Stage A/A.1 rows are not replaced. There is no provider adapter in B1. Any future external processing must explicitly enforce `DocumentStore.is_provider_processing_eligible`; an absent decision means denied. Candidate/review data and approved gold exports remain under ignored `data/private/`; no research content is intended for Git.

The OpenAI-compatible adapter and its hard request/cost budget enforcement are not implemented in B1. The approved future pilot limits (50 attempts and $1 estimated cost per run) must stop new requests when either is reached. B1 made no provider calls and needs no API credential.

Install the additional schema dependency from the updated project requirements. No API credentials or provider settings are needed for Stage B1. Run the deterministic tests with:

```powershell
python -m pytest
```

See [docs/benchmark-review.md](docs/benchmark-review.md) for the correction JSON format, review safeguards, and evidence-validation rules.

## Initial live experiment (2026-11-24)

- Tavily requests: 3, returning 30 search results.
- PDF candidates queued: 16 unique URLs from 22 discovery sightings (6 duplicate URL sightings).
- Downloads: 9 succeeded; 7 were refused by website `robots.txt` rules. No access restrictions were bypassed.
- Duplicate PDF contents detected by SHA-256: 0.
- Automated tests: 12 passed.

Eight search results did not become queued PDF candidates; their individual URLs and rejection reasons were not retained in the collection database. Search results may also be off-topic, so review document titles and sources before relying on them.
