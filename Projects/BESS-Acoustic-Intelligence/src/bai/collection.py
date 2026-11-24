import logging
import shutil
import time
from pathlib import Path

import requests

from bai.config import PROJECT_ROOT
from bai.crawler import LandingPageCrawler
from bai.database import Database
from bai.downloader import PDFDownloader
from bai.models import Candidate, SearchResult, Settings
from bai.providers.base import (
    FatalSearchProviderError,
    SearchProvider,
    SearchProviderError,
    TransientSearchError,
)
from bai.web import PoliteWebClient


logger = logging.getLogger(__name__)


class ResearchCollector:
    def __init__(self, database: Database, settings: Settings):
        self.database = database
        self.settings = settings
        self.web = PoliteWebClient(
            timeout=settings.timeout_seconds,
            rate_limit_seconds=settings.rate_limit_seconds,
        )
        self.crawler = LandingPageCrawler(
            self.web,
            max_depth=settings.crawl_depth,
            max_pages=settings.max_crawl_pages,
        )
        self.downloader = PDFDownloader(
            database,
            self.web,
            resolve_project_path(settings.download_dir),
            max_retries=settings.max_retries,
        )

    def discover(
        self,
        provider: SearchProvider | None,
        queries: list[str],
        *,
        search_depth: str,
        max_results: int,
        max_api_requests: int,
        seeds: list[Candidate] | None = None,
        dry_run: bool = False,
    ) -> tuple[list[Candidate], int]:
        candidates: list[Candidate] = []
        failures = 0

        for seed in seeds or []:
            candidates.extend(self._expand(seed))

        api_calls = 0
        fatal_provider_error = False
        for query in queries:
            if api_calls >= max_api_requests:
                logger.warning(
                    "Search request limit reached (%s); remaining queries were skipped",
                    max_api_requests,
                )
                break
            if provider is None:
                raise ValueError("A search provider is required when search queries are enabled")

            attempts = 0
            while attempts < max(1, self.settings.max_retries):
                attempts += 1
                api_calls += 1
                request_id = self.database.record_api_request(provider.name, query)
                try:
                    results = provider.search(
                        query,
                        search_depth=search_depth,
                        max_results=max_results,
                    )
                except (TransientSearchError, requests.RequestException) as exc:
                    self.database.finish_api_request(
                        request_id, "failed", error_message=str(exc)
                    )
                    logger.error("Search failed for query %r: %s", query, exc)
                    failures += 1
                    if attempts < max(1, self.settings.max_retries) and api_calls < max_api_requests:
                        time.sleep(min(2 ** (attempts - 1), 8))
                        continue
                    break
                except FatalSearchProviderError as exc:
                    self.database.finish_api_request(
                        request_id, "failed", error_message=str(exc)
                    )
                    logger.error("Stopping search after provider error: %s", exc)
                    failures += 1
                    fatal_provider_error = True
                    break
                except (SearchProviderError, RuntimeError, ValueError, OSError) as exc:
                    self.database.finish_api_request(
                        request_id, "failed", error_message=str(exc)
                    )
                    logger.error("Search failed for query %r: %s", query, exc)
                    failures += 1
                    break

                self.database.finish_api_request(
                    request_id, "succeeded", result_count=len(results)
                )
                for result in results:
                    candidates.extend(self._expand(self._as_candidate(result, query, provider.name)))
                break

            if fatal_provider_error:
                break
            if api_calls < max_api_requests:
                time.sleep(self.settings.rate_limit_seconds)

        unique_candidates = _deduplicate_candidates(candidates)
        if not dry_run:
            added, duplicates = self.database.queue_candidates(candidates)
            logger.info("Queued %s new URLs; skipped %s duplicate URLs", added, duplicates)
        return unique_candidates, failures

    def download_pending(
        self, *, limit: int | None = None, dry_run: bool = False
    ) -> tuple[int, int]:
        documents = self.database.pending_documents()
        if limit is not None:
            documents = documents[:limit]
        completed = 0
        failures = 0
        for document in documents:
            if dry_run:
                print(f"[{document['id']}] {document['source_url']} — {document['title']}")
                continue
            try:
                result = self.downloader.download(int(document["id"]))
                print(
                    f"[{document['id']}] {result.status}: "
                    f"{document['source_url']}"
                )
                completed += 1
            except RuntimeError as exc:
                logger.error("%s", exc)
                failures += 1
        return completed, failures

    def import_url(self, url: str, title: str = "") -> tuple[int, bool]:
        candidate = Candidate(
            url=url.strip(),
            title=title.strip(),
            query="manual URL import",
            provider="manual",
        )
        return self.database.add_candidate(candidate)

    def import_local_pdf(self, source: Path) -> tuple[int, bool]:
        source = source.expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"Local PDF does not exist: {source}")
        if not _is_pdf(source):
            raise ValueError(f"Local file is not a valid PDF: {source}")

        from bai.downloader import _safe_stem, _sha256

        digest = _sha256(source)
        size = source.stat().st_size
        existing = self._find_hash(digest)
        if existing:
            destination = Path(existing["local_path"])
            duplicate_of = int(existing["id"])
        else:
            destination_dir = resolve_project_path(self.settings.download_dir)
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / f"local-{digest[:16]}-{_safe_stem(source.stem, source.name)}.pdf"
            if source != destination.resolve():
                shutil.copy2(source, destination)
            duplicate_of = None

        try:
            source_url = source.as_uri()
        except ValueError as exc:
            raise ValueError(f"Cannot represent local file path as a URI: {source}") from exc
        canonical_url = f"local:{source.as_posix().casefold()}:{digest}"
        return self.database.add_local_document(
            source_url=source_url,
            canonical_url=canonical_url,
            title=source.stem,
            local_path=str(destination),
            file_size=size,
            sha256=digest,
            duplicate_of=duplicate_of,
        )

    def _find_hash(self, digest: str):
        with self.database.connect() as connection:
            return connection.execute(
                """
                SELECT * FROM documents WHERE sha256 = ?
                AND download_status IN ('downloaded', 'duplicate_content', 'imported')
                ORDER BY id LIMIT 1
                """,
                (digest,),
            ).fetchone()

    def _expand(self, candidate: Candidate) -> list[Candidate]:
        return self.crawler.discover(candidate)

    @staticmethod
    def _as_candidate(result: SearchResult, query: str, provider: str) -> Candidate:
        return Candidate(
            url=result.url,
            title=result.title,
            query=query,
            provider=provider,
        )


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _deduplicate_candidates(candidates: list[Candidate]) -> list[Candidate]:
    unique: dict[str, Candidate] = {}
    from bai.database import canonicalize_url

    for candidate in candidates:
        try:
            key = canonicalize_url(candidate.url)
        except ValueError:
            logger.warning("Skipping unsupported discovered URL: %s", candidate.url)
            continue
        previous = unique.get(key)
        if previous is None:
            unique[key] = candidate
        elif previous.title == "" and candidate.title:
            unique[key] = candidate
    return list(unique.values())


def _is_pdf(path: Path) -> bool:
    from bai.downloader import _is_pdf as validate_pdf

    return validate_pdf(path)
