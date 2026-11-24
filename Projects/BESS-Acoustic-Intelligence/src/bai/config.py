import os
import tomllib
from pathlib import Path

from dotenv import load_dotenv

from bai.database import canonicalize_url
from bai.models import Candidate, Settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUERY_FILE = PROJECT_ROOT / "config" / "search_queries.toml"
DEFAULT_SEED_FILE = PROJECT_ROOT / "config" / "seed_urls.toml"


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings(
        database_path=os.getenv("BAI_DATABASE_PATH", "data/processed/bai.sqlite3"),
        download_dir=os.getenv("BAI_DOWNLOAD_DIR", "data/raw"),
        max_api_requests_per_run=_int_env("BAI_MAX_API_REQUESTS_PER_RUN", 20),
        max_results=_int_env("BAI_MAX_RESULTS", 10),
        search_depth=os.getenv("BAI_SEARCH_DEPTH", "basic").lower(),
        timeout_seconds=_float_env("BAI_TIMEOUT_SECONDS", 20.0),
        max_retries=_int_env("BAI_MAX_RETRIES", 3),
        rate_limit_seconds=_float_env("BAI_RATE_LIMIT_SECONDS", 1.0),
        crawl_depth=_int_env("BAI_CRAWL_DEPTH", 1),
        max_crawl_pages=_int_env("BAI_MAX_CRAWL_PAGES", 25),
    )


def load_queries(path: Path = DEFAULT_QUERY_FILE) -> list[tuple[str, str]]:
    data = _load_toml(path)
    queries = data.get("queries", [])
    if not isinstance(queries, list):
        raise ValueError(f"{path} must contain [[queries]] entries")
    parsed: list[tuple[str, str]] = []
    for item in queries:
        query = item.get("query") if isinstance(item, dict) else None
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Each query in {path} must have a non-empty query value")
        parsed.append((query.strip(), str(item.get("region", "unspecified"))))
    return parsed


def load_seed_candidates(path: Path = DEFAULT_SEED_FILE) -> list[Candidate]:
    data = _load_toml(path)
    seeds = data.get("seeds", [])
    if not isinstance(seeds, list):
        raise ValueError(f"{path} must contain [[seeds]] entries")
    candidates: list[Candidate] = []
    for item in seeds:
        if not isinstance(item, dict):
            raise ValueError(f"Each seed in {path} must be a table")
        url = item.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ValueError(f"Each seed in {path} must have a non-empty URL")
        try:
            canonicalize_url(url)
        except ValueError as exc:
            raise ValueError(f"Invalid seed URL in {path}: {exc}") from exc
        candidates.append(
            Candidate(
                url=url.strip(),
                title=str(item.get("title") or ""),
                query="manual seed",
                provider="seed",
            )
        )
    return candidates


def _load_toml(path: Path) -> dict:
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
    except OSError as exc:
        raise ValueError(f"Cannot read configuration file {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML in {path}: {exc}") from exc
    return data


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be zero or greater")
    return value


def _float_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value < 0:
        raise ValueError(f"{name} must be zero or greater")
    return value
