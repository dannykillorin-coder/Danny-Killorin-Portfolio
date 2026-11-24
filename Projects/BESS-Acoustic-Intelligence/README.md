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

## Collection and provenance

The database stores each document's original URL, referring page when available, title, discovery and download timestamps, file size, SHA-256, local path, download status, and duplicate relationship. Search query/provider attribution is stored separately so multiple discovery paths are retained for a URL. Duplicate URLs are collapsed before download; identical PDF contents are detected by hash after transfer.

This is an initial discovery and collection tool, not a guarantee of search completeness or a substitute for checking each source's terms and applicability.

## Initial live experiment (2026-11-24)

- Tavily requests: 3, returning 30 search results.
- PDF candidates queued: 16 unique URLs from 22 discovery sightings (6 duplicate URL sightings).
- Downloads: 9 succeeded; 7 were refused by website `robots.txt` rules. No access restrictions were bypassed.
- Duplicate PDF contents detected by SHA-256: 0.
- Automated tests: 12 passed.

Eight search results did not become queued PDF candidates; their individual URLs and rejection reasons were not retained in the collection database. Search results may also be off-topic, so review document titles and sources before relying on them.
