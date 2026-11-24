from dataclasses import dataclass
from datetime import datetime, timezone
import math


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Candidate:
    url: str
    title: str
    query: str
    provider: str
    referring_url: str | None = None


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    content: str = ""


@dataclass(frozen=True)
class DownloadResult:
    status: str
    local_path: str | None
    file_size: int | None
    sha256: str | None
    duplicate_of: int | None = None


@dataclass(frozen=True)
class Settings:
    database_path: str = "data/processed/bai.sqlite3"
    download_dir: str = "data/raw"
    max_api_requests_per_run: int = 20
    max_results: int = 10
    search_depth: str = "basic"
    timeout_seconds: float = 20.0
    max_retries: int = 3
    rate_limit_seconds: float = 1.0
    crawl_depth: int = 1
    max_crawl_pages: int = 25

    def __post_init__(self) -> None:
        if self.max_api_requests_per_run < 0:
            raise ValueError("max_api_requests_per_run must be zero or greater")
        if not 1 <= self.max_results <= 20:
            raise ValueError("max_results must be between 1 and 20")
        if self.search_depth not in {"basic", "advanced", "fast", "ultra-fast"}:
            raise ValueError(
                "search_depth must be 'basic', 'advanced', 'fast', or 'ultra-fast'"
            )
        if not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 120:
            raise ValueError("timeout_seconds must be greater than zero and at most 120")
        if not 1 <= self.max_retries <= 10:
            raise ValueError("max_retries must be between 1 and 10")
        if not math.isfinite(self.rate_limit_seconds) or self.rate_limit_seconds < 0:
            raise ValueError("rate_limit_seconds must be zero or greater")
        if not 0 <= self.crawl_depth <= 5:
            raise ValueError("crawl_depth must be between 0 and 5")
        if not 1 <= self.max_crawl_pages <= 500:
            raise ValueError("max_crawl_pages must be between 1 and 500")
