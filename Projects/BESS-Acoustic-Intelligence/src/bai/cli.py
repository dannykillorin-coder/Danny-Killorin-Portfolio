import argparse
from dataclasses import replace
import json
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
from bai.engineering.benchmark import BenchmarkGenerator
from bai.engineering.benchmark_store import BenchmarkStore
from bai.engineering.semantic_benchmark import SemanticBenchmarkGenerator
from bai.engineering.schemas import AcousticMeasurement, BenchmarkCorrection
from bai.models import Settings
from bai.providers.tavily import TavilyProvider


logger = logging.getLogger("bai")
DEFAULT_BENCHMARK_PATH = Path(__file__).resolve().parents[2] / "data" / "private" / "benchmarks.sqlite3"


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
    benchmark = subparsers.add_parser(
        "benchmark",
        help="Generate and manually review local engineering benchmark candidates",
    )
    benchmark_commands = benchmark.add_subparsers(
        dest="benchmark_command",
        required=True,
    )
    generate_benchmark = benchmark_commands.add_parser(
        "generate",
        help="Generate deterministic, unapproved candidates from parsed tables",
    )
    generate_benchmark.add_argument("--limit", type=int, default=1000)
    list_benchmark = benchmark_commands.add_parser(
        "list",
        help="List benchmark candidates, including existing review items",
    )
    list_benchmark.add_argument(
        "--status",
        choices=("pending", "rejected", "approved"),
        help="Filter by review status; defaults to all statuses",
    )
    list_benchmark.add_argument(
        "--generation-version",
        choices=("stage-b1.0", "stage-b1.1"),
        help="Filter by candidate-generation version",
    )
    show_benchmark = benchmark_commands.add_parser(
        "show",
        help="Show a candidate and its exact evidence",
    )
    show_benchmark.add_argument("candidate_id")
    review_benchmark = benchmark_commands.add_parser(
        "review",
        help="Approve, reject, or append a correction for a candidate",
    )
    review_benchmark.add_argument("candidate_id")
    review_benchmark.add_argument(
        "--action",
        choices=("approve", "reject", "correct"),
        required=True,
    )
    review_benchmark.add_argument("--reviewer", required=True)
    review_benchmark.add_argument("--reason", default="")
    review_benchmark.add_argument("--correction-file", type=Path)
    review_benchmark.add_argument(
        "--comparison-eligibility",
        choices=("not_assessed", "eligible", "ineligible"),
    )
    benchmark_commands.add_parser(
        "stats",
        help="Show pending candidates, review history, and approved gold records",
    )
    benchmark_commands.add_parser(
        "reviews",
        help="List persisted open semantic and extraction review items",
    )
    compare_benchmark = benchmark_commands.add_parser(
        "compare",
        help="Compare candidate generations at matching source table cells",
    )
    compare_benchmark.add_argument(
        "--old-version",
        choices=("stage-b1.0", "stage-b1.1"),
        default="stage-b1.0",
    )
    compare_benchmark.add_argument(
        "--new-version",
        choices=("stage-b1.0", "stage-b1.1"),
        default="stage-b1.1",
    )
    export_benchmark = benchmark_commands.add_parser(
        "export",
        help="Export only manually approved gold-standard records",
    )
    export_benchmark.add_argument(
        "--output",
        type=Path,
        default=Path("data/private/gold-standard.json"),
    )

    provider_eligibility = subparsers.add_parser(
        "provider-eligibility",
        help="Review default-deny document eligibility for future provider processing",
    )
    provider_commands = provider_eligibility.add_subparsers(
        dest="provider_eligibility_command",
        required=True,
    )
    provider_commands.add_parser("list", help="List provider eligibility for all documents")
    set_eligibility = provider_commands.add_parser(
        "set",
        help="Record an explicit provider processing allow or deny decision",
    )
    set_eligibility.add_argument("document_id", type=int)
    decision = set_eligibility.add_mutually_exclusive_group(required=True)
    decision.add_argument("--allow", action="store_true")
    decision.add_argument("--deny", action="store_true")
    set_eligibility.add_argument("--reviewer", required=True)
    set_eligibility.add_argument("--basis", required=True)
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
        if args.command == "benchmark":
            return _benchmark(args, database)
        if args.command == "provider-eligibility":
            return _provider_eligibility(args, DocumentStore(database))
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


def _benchmark(args, database: Database) -> int:
    store = BenchmarkStore(DEFAULT_BENCHMARK_PATH)
    if args.benchmark_command == "generate":
        generator = SemanticBenchmarkGenerator(database, store)
        added, existing, selected = generator.generate(args.limit)
        stats = store.statistics()
        print(
            f"B1.1 selected {selected} candidate(s): {added} new, "
            f"{existing} already present; {stats['open_review_items']} open review item(s). "
            "All candidates remain unapproved."
        )
        return 0

    store.initialize()
    if args.benchmark_command == "list":
        rows = store.list_candidates(args.status, args.generation_version)
        if not rows:
            print("No benchmark candidates match the requested status.")
            return 0
        for candidate_id, status, candidate in rows:
            measurement = candidate.measurement
            value = (
                str(measurement.value)
                if measurement.value is not None
                else "unknown"
            )
            unit = measurement.units or "unit unknown"
            print(
                f"{candidate_id} [{status}] {candidate.source.title} — "
                f"page {candidate.page_number}, "
                f"{candidate.source_table_reference or 'table reference unknown'}, "
                f"{_format_measurement(measurement)} {unit}, "
                f"{measurement.metric_type}, {candidate.record_kind}, "
                f"{candidate.bess_relevance}"
            )
        return 0

    if args.benchmark_command == "show":
        candidate = store.get_candidate(args.candidate_id)
        payload = json.loads(candidate.model_dump_json(indent=2))
        current_measurement, human_evidence = store.current_review(args.candidate_id)
        payload["current_reviewed_measurement"] = json.loads(
            current_measurement.model_dump_json(indent=2)
        )
        payload["current_human_review_evidence"] = [
            json.loads(item.model_dump_json(indent=2))
            for item in human_evidence
        ]
        payload["review_history"] = [
            json.loads(event.model_dump_json(indent=2))
            for event in store.review_history(args.candidate_id)
        ]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.benchmark_command == "review":
        if not args.reviewer.strip():
            raise ValueError("--reviewer must not be empty")
        candidate = store.get_candidate(args.candidate_id)
        if args.action == "correct":
            if args.correction_file is None:
                raise ValueError("--action correct requires --correction-file")
            if not args.reason.strip():
                raise ValueError("A correction requires --reason")
            correction = BenchmarkCorrection.model_validate_json(
                args.correction_file.read_text(encoding="utf-8")
            )
            corrected = correction.measurement
            human_evidence = correction.human_review_evidence
            from bai.engineering.validation import validate_candidate_evidence

            validate_candidate_evidence(
                candidate,
                database,
                corrected,
                human_evidence,
            )
            store.record_correction(
                args.candidate_id,
                corrected,
                args.reviewer,
                args.reason,
                human_evidence,
            )
            print(
                f"Correction recorded for {args.candidate_id}; original candidate "
                "and correction history were preserved."
            )
            return 0
        if args.correction_file is not None:
            raise ValueError("--correction-file is only valid with --action correct")
        if args.action == "reject":
            if not args.reason.strip():
                raise ValueError("A rejection requires --reason")
            store.reject(args.candidate_id, args.reviewer, args.reason)
            print(f"Candidate {args.candidate_id} rejected; review history preserved.")
            return 0

        from bai.engineering.validation import validate_candidate_evidence

        measurement, human_evidence = store.current_review(args.candidate_id)
        measurement_data = measurement.model_dump()
        if args.comparison_eligibility is not None:
            measurement_data["comparison_eligibility"] = args.comparison_eligibility
        measurement = AcousticMeasurement.model_validate(measurement_data)
        validate_candidate_evidence(
            candidate,
            database,
            measurement,
            human_evidence,
        )
        record = store.approve(
            args.candidate_id,
            args.reviewer,
            args.reason,
            measurement,
            human_evidence,
        )
        print(
            f"Candidate {args.candidate_id} approved into the separate gold-standard "
            f"set by {record.approved_by}; comparison eligibility is "
            f"{record.measurement.comparison_eligibility}."
        )
        return 0

    if args.benchmark_command == "stats":
        stats = store.statistics()
        for label in (
            "pending",
            "rejected",
            "approved",
            "gold_records",
            "review_events",
            "stage-b1.0",
            "stage-b1.1",
            "review_items",
            "open_review_items",
        ):
            print(f"{label.replace('_', ' ').title()}: {stats[label]}")
        return 0

    if args.benchmark_command == "reviews":
        items = store.review_items("open")
        if not items:
            print("No open benchmark review items.")
            return 0
        for item in items:
            table = item.source_table_reference or f"table index {item.table_index}"
            location = ""
            if item.row_index is not None and item.column_index is not None:
                location = f", source cell ({item.row_index}, {item.column_index})"
            print(
                f"{item.review_item_id} [{item.generator_version}] "
                f"{item.source.title}, page {item.page_number}, {table}{location}: "
                f"{item.reason}"
            )
        return 0

    if args.benchmark_command == "compare":
        _compare_benchmark_generations(
            store,
            database,
            args.old_version,
            args.new_version,
        )
        return 0

    if args.benchmark_command == "export":
        output = _private_benchmark_output(args.output)
        records = store.gold_records()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                [json.loads(record.model_dump_json()) for record in records],
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Exported {len(records)} manually approved record(s) to {output}")
        return 0

    raise ValueError(f"Unsupported benchmark command: {args.benchmark_command}")


def _provider_eligibility(args, store: DocumentStore) -> int:
    store.initialize()
    if args.provider_eligibility_command == "list":
        for row in store.provider_eligibility():
            state = "eligible" if row["eligible"] else "DENIED (default)"
            if row["recorded_at"] and not row["eligible"]:
                state = "denied"
            print(
                f"Document {row['document_id']}: {state} — {row['title'] or row['source_url']}"
            )
            if row["reviewer"]:
                print(
                    f"  decision by {row['reviewer']} on {row['recorded_at']}: "
                    f"{row['authorization_basis']}"
                )
        return 0
    store.set_provider_eligibility(
        args.document_id,
        args.allow,
        args.reviewer,
        args.basis,
    )
    state = "allowed" if args.allow else "denied"
    print(f"Recorded provider processing as {state} for document {args.document_id}.")
    return 0


def _private_benchmark_output(path: Path) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    private_root = (project_root / "data" / "private").resolve()
    target = path if path.is_absolute() else project_root / path
    target = target.resolve()
    try:
        target.relative_to(private_root)
    except ValueError as exc:
        raise ValueError("Benchmark exports must remain inside data/private") from exc
    return target


def _format_measurement(measurement: AcousticMeasurement) -> str:
    if measurement.comparator == "exact" and measurement.value is not None:
        return str(measurement.value)
    if measurement.comparator in {"lt", "lte"} and measurement.upper_bound is not None:
        return ("<" if measurement.comparator == "lt" else "≤") + str(
            measurement.upper_bound
        )
    if measurement.comparator in {"gt", "gte"} and measurement.lower_bound is not None:
        return (">" if measurement.comparator == "gt" else "≥") + str(
            measurement.lower_bound
        )
    if (
        measurement.comparator == "range"
        and measurement.lower_bound is not None
        and measurement.upper_bound is not None
    ):
        return f"{measurement.lower_bound}–{measurement.upper_bound}"
    if measurement.value is not None:
        return f"{measurement.value} (legacy numeric interpretation)"
    return measurement.source_value_text


def _compare_benchmark_generations(
    store: BenchmarkStore,
    database: Database,
    old_version: str,
    new_version: str,
) -> None:
    if old_version == new_version:
        raise ValueError("Choose different generation versions to compare")
    rows_by_version = {
        version: store.list_candidates(generator_version=version)
        for version in (old_version, new_version)
    }
    table_ids = {
        candidate.table_id
        for rows in rows_by_version.values()
        for _, _, candidate in rows
    }
    table_indices = {}
    if table_ids:
        placeholders = ",".join("?" for _ in table_ids)
        with database.connect() as connection:
            table_indices = {
                int(table_id): int(table_index)
                for table_id, table_index in connection.execute(
                    f"SELECT id, table_index FROM extracted_tables "
                    f"WHERE id IN ({placeholders})",
                    tuple(table_ids),
                )
            }

    def keyed(rows):
        result = {}
        for _, _, candidate in rows:
            value_cells = [
                item
                for item in candidate.evidence
                if item.role == "measurement_value"
            ]
            for citation in value_cells:
                if citation.row_index is None or citation.column_index is None:
                    continue
                key = (
                    candidate.source.sha256,
                    candidate.page_number,
                    table_indices.get(candidate.table_id, candidate.table_id),
                    citation.row_index,
                    citation.column_index,
                )
                result.setdefault(key, []).append(candidate)
        return result

    old = keyed(rows_by_version[old_version])
    new = keyed(rows_by_version[new_version])
    locations = sorted(set(old) | set(new))
    matched = sum(key in old and key in new for key in locations)
    print(
        f"{old_version}: {len(rows_by_version[old_version])} candidate(s); "
        f"{new_version}: {len(rows_by_version[new_version])} candidate(s); "
        f"{matched} source cell(s) matched, "
        f"{sum(key in old and key not in new for key in locations)} old-only, "
        f"{sum(key in new and key not in old for key in locations)} new-only."
    )
    for key in locations:
        old_items = old.get(key, [])
        new_items = new.get(key, [])
        old_candidate = old_items[0] if old_items else None
        new_candidate = new_items[0] if new_items else None
        candidate = new_candidate or old_candidate
        if candidate is None:
            continue
        sha, page, table_index, row_index, column_index = key
        old_summary = _candidate_comparison_summary(old_candidate)
        new_summary = _candidate_comparison_summary(new_candidate)
        if old_summary == new_summary:
            continue
        print(
            f"{candidate.source.title} p{page} table-index {table_index} "
            f"cell ({row_index}, {column_index}) [{sha[:12]}]\n"
            f"  old: {old_summary or 'not generated'}\n"
            f"  new: {new_summary or 'not generated'}"
        )


def _candidate_comparison_summary(candidate) -> dict | None:
    if candidate is None:
        return None
    measurement = candidate.measurement
    return {
        "source_value": measurement.source_value_text,
        "comparator": measurement.comparator,
        "value": str(measurement.value) if measurement.value is not None else None,
        "bounds": [
            str(measurement.lower_bound) if measurement.lower_bound is not None else None,
            str(measurement.upper_bound) if measurement.upper_bound is not None else None,
        ],
        "units": measurement.units,
        "metric": measurement.metric_type,
        "record_kind": candidate.record_kind,
        "relevance": candidate.bess_relevance,
        "entity_role": candidate.source_entity_role,
        "entity": candidate.source_entity_label,
        "basis": measurement.value_basis,
        "frequency_hz": (
            str(measurement.frequency_band.center_hz)
            if measurement.frequency_band
            else None
        ),
    }


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
