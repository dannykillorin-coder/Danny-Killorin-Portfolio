from pathlib import Path

import requests
import pytest

from bai.collection import ResearchCollector
from bai.config import load_queries, load_seed_candidates
from bai.crawler import LandingPageCrawler
from bai.database import Database
from bai.downloader import PDFDownloader
from bai.models import Candidate, SearchResult, Settings
from bai.providers.tavily import TavilyProvider
from bai.providers.base import FatalSearchProviderError, TransientSearchError
from bai.web import PoliteWebClient


FIXTURES = Path(__file__).parent / "fixtures"
PDF_BYTES = (FIXTURES / "minimal.pdf.fixture").read_bytes()


class MockSearchClient:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "results": [
                {"title": "BESS assessment", "url": "https://example.org/report.pdf"},
                {"title": "No URL"},
            ]
        }


class FakeResponse:
    def __init__(self, content: bytes, *, status_code: int = 200, content_type: str = "application/pdf"):
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}
        self.content = content
        self.text = content.decode("utf-8", errors="replace")
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset : offset + chunk_size]

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class FakeWeb:
    def __init__(self, response_factory):
        self.response_factory = response_factory
        self.calls = []

    def get(self, url, *, headers=None):
        self.calls.append((url, headers or {}))
        return self.response_factory(url, headers or {})


def test_tavily_provider_maps_mocked_results_and_search_options():
    client = MockSearchClient()
    provider = TavilyProvider(client=client)

    results = provider.search("BESS acoustic", search_depth="advanced", max_results=7)

    assert results == [
        SearchResult(
            title="BESS assessment",
            url="https://example.org/report.pdf",
            content="",
        )
    ]
    assert client.calls[0]["search_depth"] == "advanced"
    assert client.calls[0]["max_results"] == 7
    assert client.calls[0]["timeout"] == 20


def test_query_configuration_has_requested_regions_and_seed_file_loads():
    queries = load_queries()
    regions = {region for _, region in queries}

    assert len(queries) >= 12
    assert {
        "United States",
        "United Kingdom",
        "Australia",
        "Canada",
        "Continental Europe",
    }.issubset(regions)
    assert load_seed_candidates() == []


def test_crawler_extracts_pdf_links_and_preserves_referrer():
    html = (FIXTURES / "landing.html").read_bytes()
    web = FakeWeb(
        lambda url, _headers: FakeResponse(
            html if url.endswith("/planning/") else b"<html></html>",
            content_type="text/html; charset=utf-8",
        )
    )
    crawler = LandingPageCrawler(web, max_depth=1, max_pages=5)

    candidates = crawler.discover(
        Candidate(
            "https://example.org/planning/",
            "Planning application",
            "BESS noise planning",
            "tavily",
        )
    )

    assert len(candidates) == 1
    assert candidates[0].url == "https://example.org/reports/noise.pdf"
    assert candidates[0].referring_url == "https://example.org/planning/"
    assert candidates[0].title == "Acoustic report"


def test_database_collapses_duplicate_urls_but_keeps_discovery_provenance(tmp_path):
    database = Database(tmp_path / "documents.sqlite3")
    first = Candidate(
        "https://EXAMPLE.org:443/report.pdf#page=1",
        "Assessment",
        "query one",
        "tavily",
    )
    second = Candidate(
        "https://example.org/report.pdf",
        "Assessment",
        "query two",
        "tavily",
    )

    document_id, is_new = database.add_candidate(first)
    duplicate_id, second_is_new = database.add_candidate(second)

    assert is_new is True
    assert second_is_new is False
    assert duplicate_id == document_id
    assert len(database.discoveries_for(document_id)) == 2
    assert database.get_document(document_id)["source_url"] == first.url


def test_download_resumes_partial_pdf_and_deduplicates_content(tmp_path):
    database = Database(tmp_path / "documents.sqlite3")
    first_id, _ = database.add_candidate(
        Candidate("https://example.org/one.pdf", "Report one", "q", "manual")
    )
    second_id, _ = database.add_candidate(
        Candidate("https://example.org/two.pdf", "Report two", "q", "manual")
    )
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    part = download_dir / f"{first_id}-Report-one.pdf.part"
    split_at = 9
    part.write_bytes(PDF_BYTES[:split_at])

    web = FakeWeb(
        lambda _url, headers: (
            _response_with_range(PDF_BYTES[split_at:], split_at)
            if headers.get("Range")
            else FakeResponse(PDF_BYTES)
        )
    )
    downloader = PDFDownloader(
        database,
        web,
        download_dir,
        max_retries=2,
    )

    first = downloader.download(first_id)
    second = downloader.download(second_id)

    assert web.calls[0][1]["Range"] == f"bytes={split_at}-"
    assert first.status == "downloaded"
    assert Path(first.local_path).read_bytes() == PDF_BYTES
    assert second.status == "duplicate_content"
    assert second.sha256 == first.sha256
    assert second.duplicate_of == first_id


def test_non_pdf_response_is_not_saved(tmp_path):
    database = Database(tmp_path / "documents.sqlite3")
    document_id, _ = database.add_candidate(
        Candidate("https://example.org/not-a-pdf", "Unexpected page", "q", "manual")
    )
    web = FakeWeb(lambda _url, _headers: FakeResponse(b"<html>not a PDF</html>"))
    downloader = PDFDownloader(
        database, web, tmp_path / "downloads", max_retries=1
    )

    try:
        downloader.download(document_id)
    except RuntimeError as exc:
        assert "not a valid PDF" in str(exc)
    else:
        raise AssertionError("Invalid content should fail PDF validation")

    assert database.get_document(document_id)["download_status"] == "failed"
    assert list((tmp_path / "downloads").iterdir()) == []


def test_local_pdf_import_records_metadata_and_detects_duplicate(tmp_path):
    database = Database(tmp_path / "documents.sqlite3")
    collector = ResearchCollector(
        database,
        Settings(
            database_path=str(tmp_path / "documents.sqlite3"),
            download_dir=str(tmp_path / "downloads"),
            rate_limit_seconds=0,
        ),
    )
    first_id, first_is_new = collector.import_local_pdf(FIXTURES / "minimal.pdf.fixture")
    second_path = tmp_path / "copy.pdf"
    second_path.write_bytes(PDF_BYTES)

    second_id, second_is_new = collector.import_local_pdf(second_path)

    assert first_is_new is True
    assert second_is_new is True
    assert database.get_document(first_id)["download_status"] == "imported"
    duplicate = database.get_document(second_id)
    assert duplicate["download_status"] == "duplicate_content"
    assert duplicate["duplicate_of_document_id"] == first_id


def test_web_client_does_not_fetch_robots_disallowed_urls():
    class RobotsOnlySession:
        def __init__(self):
            self.urls = []

        def get(self, url, **_kwargs):
            self.urls.append(url)
            response = requests.Response()
            response.status_code = 200
            response.url = url
            response._content = b"User-agent: *\nDisallow: /restricted\n"
            return response

    session = RobotsOnlySession()
    web = PoliteWebClient(
        timeout=1,
        rate_limit_seconds=0,
        session=session,
        sleeper=lambda _seconds: None,
    )

    with pytest.raises(PermissionError, match="robots.txt"):
        web.get("https://example.org/restricted/report.pdf")

    assert session.urls == ["https://example.org/robots.txt"]


def test_web_client_honors_robots_crawl_delay():
    class RobotsDelaySession:
        def get(self, url, **_kwargs):
            response = requests.Response()
            response.status_code = 200
            response.url = url
            response.headers["Content-Type"] = "application/pdf"
            response._content = (
                b"User-agent: *\nCrawl-delay: 5\n"
                if url.endswith("/robots.txt")
                else b"%PDF-1.4"
            )
            response._content_consumed = True
            return response

    delays = []
    web = PoliteWebClient(
        timeout=1,
        rate_limit_seconds=0,
        session=RobotsDelaySession(),
        sleeper=delays.append,
    )

    response = web.get("https://example.org/report.pdf")

    assert response.status_code == 200
    assert delays and delays[0] >= 4
    response.close()


def test_api_request_limit_counts_and_stops_before_extra_queries(tmp_path):
    database = Database(tmp_path / "documents.sqlite3")
    settings = Settings(
        database_path=str(tmp_path / "documents.sqlite3"),
        download_dir=str(tmp_path / "downloads"),
        max_api_requests_per_run=1,
        max_results=5,
        rate_limit_seconds=0,
    )
    collector = ResearchCollector(database, settings)

    class Provider:
        name = "mock"

        def __init__(self):
            self.queries = []

        def search(self, query, *, search_depth, max_results):
            self.queries.append(query)
            return [SearchResult("A report", "https://example.org/a.pdf")]

    provider = Provider()
    candidates, failures = collector.discover(
        provider,
        ["first query", "second query"],
        search_depth="basic",
        max_results=5,
        max_api_requests=1,
    )

    assert [query for query in provider.queries] == ["first query"]
    assert len(candidates) == 1
    assert failures == 0
    assert database.request_count() == 1
    assert database.statistics()["discovered"] == 1


def test_transient_search_retries_are_counted_against_request_limit(tmp_path):
    database = Database(tmp_path / "documents.sqlite3")
    settings = Settings(
        database_path=str(tmp_path / "documents.sqlite3"),
        download_dir=str(tmp_path / "downloads"),
        max_retries=5,
        rate_limit_seconds=0,
    )
    collector = ResearchCollector(database, settings)

    class Provider:
        name = "mock"

        def __init__(self):
            self.calls = 0

        def search(self, _query, *, search_depth, max_results):
            self.calls += 1
            raise TransientSearchError("temporary network error")

    provider = Provider()
    _, failures = collector.discover(
        provider,
        ["query"],
        search_depth="basic",
        max_results=5,
        max_api_requests=2,
    )

    assert provider.calls == 2
    assert failures == 2
    assert database.request_count() == 2
    assert [row["status"] for row in database.api_request_rows()] == ["failed", "failed"]


def test_fatal_provider_error_stops_remaining_queries(tmp_path):
    database = Database(tmp_path / "documents.sqlite3")
    collector = ResearchCollector(
        database,
        Settings(
            database_path=str(tmp_path / "documents.sqlite3"),
            download_dir=str(tmp_path / "downloads"),
            rate_limit_seconds=0,
        ),
    )

    class Provider:
        name = "mock"

        def __init__(self):
            self.calls = 0

        def search(self, _query, *, search_depth, max_results):
            self.calls += 1
            raise FatalSearchProviderError("usage limit reached")

    provider = Provider()
    _, failures = collector.discover(
        provider,
        ["query one", "query two"],
        search_depth="basic",
        max_results=5,
        max_api_requests=10,
    )

    assert provider.calls == 1
    assert failures == 1
    assert database.request_count() == 1


def _response_with_range(content: bytes, start: int) -> FakeResponse:
    response = FakeResponse(content, status_code=206)
    response.headers["Content-Range"] = f"bytes {start}-{len(PDF_BYTES) - 1}/{len(PDF_BYTES)}"
    return response
