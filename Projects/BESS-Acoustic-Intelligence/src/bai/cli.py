import argparse
from dataclasses import replace
import logging
import sqlite3
import sys
from pathlib import Path

from bai.collection import ResearchCollector, resolve_project_path
from bai.config import (
    DEFAULT_QUERY_FILE,
    DEFAULT_SEED_FILE,
    load_queries,
    load_seed_candidates,
    load_settings,
)
from bai.database import Database
from bai.documents.pipeline import DocumentIntelligence
from bai.documents.store import DocumentStore
from bai.models import Settings
from bai.providers.tavily import TavilyProvider


logger = logging.getLogger("bai")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bai",
        description="Discover, collect, and analyze publicly accessible BESS acoustic PDFs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover", help="Search, inspect, and queue PDF URLs")
    discover.add_argument("--query-file", type=Path, default=DEFAULT_QUERY_FILE)
    discover.add_argument("--seed-file", type=Path, default=DEFAULT_SEED_FILE)
    discover.add_argument("--no-seeds", action="store_true", help="Do not process configured seed URLs")
    discover.add_argument("--query", action="append", help="Override configured queries; repeat as needed")
    discover.add_argument("--max-results", type=int, help="Results per Tavily query (1-20)")
    discover.add_argument(
        "--search-depth",
        choices=("basic", "advanced", "fast", "ultra-fast"),
        help="Tavily search depth",
    )
    discover.add_argument(
        "--max-api-requests",
        type=int,
        help="Maximum Tavily requests in this invocation (including retries)",
    )
    discover.add_argument(
        "--crawl-depth",
        type=int,
        help="Maximum same-site landing-page link depth (0-5)",
    )
    discover.add_argument(
        "--max-crawl-pages",
        type=int,
        help="Maximum landing pages inspected per search result/seed",
    )
    discover.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered URLs without queueing documents (API calls are still made)",
    )

    download = subparsers.add_parser("download", help="Download queued PDFs")
    download.add_argument("--limit", type=int, help="Maximum documents to process")
    download.add_argument(
        "--dry-run",
        action="store_true",
        help="List queued URLs without downloading them",
    )

    import_url = subparsers.add_parser("import-url", help="Queue a manually supplied PDF URL")
    import_url.add_argument("url")
    import_url.add_argument("--title", default="")

    import_file = subparsers.add_parser("import-file", help="Validate and import a local PDF")
    import_file.add_argument("path", type=Path)

    subparsers.add_parser("stats", help="Show collection and API request statistics")

    parse = subparsers.add_parser(
        "parse",
        help="Extract text, candidate tables, and acoustic page signals from collected PDFs",
    )
    parse.add_argument("--limit", type=int, help="Maximum documents to process")
    parse.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the number of uncached documents without parsing them",
    )

    classify = subparsers.add_parser(
        "classify",
        help="Classify parsed documents using deterministic evidence",
    )
    classify.add_argument("--limit", type=int, help="Maximum documents to classify")
    classify.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the number of unclassified documents",
    )

    subparsers.add_parser(
        "intelligence-stats",
        help="Show PDF processing, classification, and review statistics",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings()
    _configure_logging()

    try:
        _validate_settings(settings)
        database = Database(resolve_project_path(settings.database_path))
        if args.command == "discover":
            return _discover(args, settings, database)
        if args.command == "download":
            return _download(args, ResearchCollector(database, settings))
        if args.command == "import-url":
            collector = ResearchCollector(database, settings)
            document_id, is_new = collector.import_url(args.url, args.title)
            print(f"{'Queued' if is_new else 'Already known'} document [{document_id}]: {args.url}")
            return 0
        if args.command == "import-file":
            collector = ResearchCollector(database, settings)
            document_id, is_new = collector.import_local_pdf(args.path)
            print(f"{'Imported' if is_new else 'Already imported'} local PDF [{document_id}]")
            return 0
        if args.command == "stats":
            _print_stats(database)
            return 0
        if args.command == "parse":
            return _parse_documents(args, database)
        if args.command == "classify":
            return _classify_documents(args, database)
        if args.command == "intelligence-stats":
            _print_intelligence_stats(DocumentStore(database))
            return 0
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        logger.error("%s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unsupported command {args.command}")
    return 2


def _discover(args, settings: Settings, database: Database) -> int:
    if args.max_results is not None and not 1 <= args.max_results <= 20:
        raise ValueError("--max-results must be between 1 and 20 (Tavily limit)")
    if args.max_api_requests is not None and args.max_api_requests < 0:
        raise ValueError("--max-api-requests must be zero or greater")
    if args.crawl_depth is not None and not 0 <= args.crawl_depth <= 5:
        raise ValueError("--crawl-depth must be between 0 and 5")
    if args.max_crawl_pages is not None and not 1 <= args.max_crawl_pages <= 500:
        raise ValueError("--max-crawl-pages must be between 1 and 500")

    settings = replace(
        settings,
        crawl_depth=args.crawl_depth if args.crawl_depth is not None else settings.crawl_depth,
        max_crawl_pages=(
            args.max_crawl_pages
            if args.max_crawl_pages is not None
            else settings.max_crawl_pages
        ),
    )
    configured = load_queries(args.query_file)
    queries = args.query if args.query else [query for query, _ in configured]
    queries = [query.strip() for query in queries if query.strip()]
    if not queries and (args.no_seeds or not load_seed_candidates(args.seed_file)):
        raise ValueError("No queries or seed URLs are configured")
    seeds = [] if args.no_seeds else load_seed_candidates(args.seed_file)
    max_requests = (
        args.max_api_requests
        if args.max_api_requests is not None
        else settings.max_api_requests_per_run
    )
    max_results = args.max_results if args.max_results is not None else settings.max_results
    search_depth = args.search_depth if args.search_depth is not None else settings.search_depth
    provider = (
        TavilyProvider(timeout_seconds=settings.timeout_seconds)
        if queries and max_requests > 0
        else None
    )
    collector = ResearchCollector(database, settings)
    candidates, failures = collector.discover(
        provider,
        queries,
        search_depth=search_depth,
        max_results=max_results,
        max_api_requests=max_requests,
        seeds=seeds,
        dry_run=args.dry_run,
    )
    label = "DRY RUN — " if args.dry_run else ""
    print(f"{label}Discovered {len(candidates)} unique PDF URL(s); search failures: {failures}")
    for candidate in candidates:
        referring = f" (from {candidate.referring_url})" if candidate.referring_url else ""
        print(f"- {candidate.url}{referring}")
    if args.dry_run:
        print("No document URLs were queued. Search API calls were recorded.")
    elif candidates:
        print("Use `python -m bai download --dry-run` to review queued URLs before downloading.")
    return 1 if failures else 0


def _download(args, collector: ResearchCollector) -> int:
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be zero or greater")
    completed, failures = collector.download_pending(
        limit=args.limit,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print("Dry run complete; no files were downloaded.")
    else:
        print(f"Download processing complete: {completed} completed, {failures} failed.")
    return 1 if failures else 0


def _print_stats(database: Database) -> None:
    stats = database.statistics()
    print(f"Documents: {stats.pop('documents')}")
    print(f"Tracked search API requests: {stats.pop('api_requests')}")
    for status, count in sorted(stats.items()):
        print(f"{status}: {count}")


def _parse_documents(args, database: Database) -> int:
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be zero or greater")
    summary = DocumentIntelligence(database).parse_documents(
        limit=args.limit,
        dry_run=args.dry_run,
    )
    prefix = "Would process" if args.dry_run else "Processed"
    print(
        f"{prefix} {summary.processed} document(s); skipped {summary.skipped}; "
        f"failed {summary.failed}; review required {summary.review_required}; "
        f"outstanding review items {summary.outstanding_review}."
    )
    return 1 if summary.failed and not args.dry_run else 0


def _classify_documents(args, database: Database) -> int:
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be zero or greater")
    summary = DocumentIntelligence(database).classify_documents(
        limit=args.limit,
        dry_run=args.dry_run,
    )
    prefix = "Would classify" if args.dry_run else "Classified"
    print(
        f"{prefix} {summary.processed} document(s); skipped {summary.skipped}; "
        f"review required {summary.review_required}; "
        f"outstanding review items {summary.outstanding_review}."
    )
    return 0


def _print_intelligence_stats(store: DocumentStore) -> None:
    store.initialize()
    stats = store.statistics()
    print("Document intelligence:")
    for label, key in (
        ("Documents with latest processing results", "latest_processed_documents"),
        ("Pages in latest results", "latest_parsed_pages"),
        ("Tables in latest results", "latest_candidate_tables"),
        ("Successfully extracted tables", "successfully_extracted_tables"),
        ("Missing-text pages", "missing_text_pages"),
        ("Parsing failures", "parse_failures"),
        ("Ambiguous tables", "ambiguous_tables"),
        ("Tables with unspecified measurement type", "missing_measurement_type_tables"),
        ("Documents flagged for review", "review_required_documents"),
        ("Outstanding review items", "outstanding_review_items"),
        ("Historical processing runs", "processing_runs"),
        ("Historical extracted pages", "parsed_pages"),
        ("Historical candidate tables", "candidate_tables"),
        ("Historical classification records", "classifications"),
        ("Historical source evidence records", "source_evidence"),
    ):
        print(f"{label}: {stats[key]}")


def _validate_settings(settings: Settings) -> None:
    if settings.search_depth not in {"basic", "advanced", "fast", "ultra-fast"}:
        raise ValueError(
            "BAI_SEARCH_DEPTH must be 'basic', 'advanced', 'fast', or 'ultra-fast'"
        )
    if not 1 <= settings.max_results <= 20:
        raise ValueError("BAI_MAX_RESULTS must be between 1 and 20")
    if settings.timeout_seconds <= 0:
        raise ValueError("BAI_TIMEOUT_SECONDS must be greater than zero")
    if settings.max_retries < 1:
        raise ValueError("BAI_MAX_RETRIES must be at least one")
    if settings.crawl_depth > 5:
        raise ValueError("BAI_CRAWL_DEPTH must not exceed 5")
    if settings.max_crawl_pages > 500:
        raise ValueError("BAI_MAX_CRAWL_PAGES must not exceed 500")


def _configure_logging() -> None:
    log_path = resolve_project_path("logs/bai.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
