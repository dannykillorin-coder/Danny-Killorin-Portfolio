import sqlite3
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterator

from bai.models import Candidate, DownloadResult, utc_now


def canonicalize_url(url: str) -> str:
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"Only absolute HTTP(S) URLs are supported: {url}")
    if parts.username or parts.password:
        raise ValueError("URLs containing embedded credentials are not allowed")
    scheme = parts.scheme.lower()
    hostname = parts.hostname.lower()
    port = parts.port
    netloc = hostname if port is None or (scheme, port) in {("http", 80), ("https", 443)} else f"{hostname}:{port}"
    path = parts.path or "/"
    return urlunsplit((scheme, netloc, path, parts.query, ""))


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    canonical_url TEXT NOT NULL UNIQUE,
                    source_url TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    referring_url TEXT,
                    discovery_timestamp TEXT NOT NULL,
                    download_timestamp TEXT,
                    local_path TEXT,
                    file_size INTEGER,
                    sha256 TEXT,
                    download_status TEXT NOT NULL DEFAULT 'discovered',
                    error_message TEXT,
                    duplicate_of_document_id INTEGER REFERENCES documents(id),
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS discoveries (
                    id INTEGER PRIMARY KEY,
                    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    search_query TEXT NOT NULL,
                    search_provider TEXT NOT NULL,
                    referring_url TEXT NOT NULL DEFAULT '',
                    discovery_timestamp TEXT NOT NULL,
                    UNIQUE(document_id, search_query, search_provider, referring_url)
                );
                CREATE TABLE IF NOT EXISTS api_requests (
                    id INTEGER PRIMARY KEY,
                    provider TEXT NOT NULL,
                    search_query TEXT NOT NULL,
                    requested_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    result_count INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(download_status);
                CREATE INDEX IF NOT EXISTS idx_documents_sha256 ON documents(sha256);
                CREATE INDEX IF NOT EXISTS idx_discoveries_document ON discoveries(document_id);
                """
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def add_candidate(self, candidate: Candidate) -> tuple[int, bool]:
        canonical_url = canonicalize_url(candidate.url)
        discovered = utc_now()
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id, title FROM documents WHERE canonical_url = ?",
                (canonical_url,),
            ).fetchone()
            is_new = row is None
            if is_new:
                cursor = connection.execute(
                    """
                    INSERT INTO documents (
                        canonical_url, source_url, title, referring_url,
                        discovery_timestamp, download_status, created_at
                    ) VALUES (?, ?, ?, ?, ?, 'discovered', ?)
                    """,
                    (
                        canonical_url,
                        candidate.url,
                        candidate.title.strip(),
                        candidate.referring_url,
                        discovered,
                        discovered,
                    ),
                )
                document_id = int(cursor.lastrowid)
            else:
                document_id = int(row["id"])
                if not row["title"] and candidate.title.strip():
                    connection.execute(
                        "UPDATE documents SET title = ? WHERE id = ?",
                        (candidate.title.strip(), document_id),
                    )

            connection.execute(
                """
                INSERT OR IGNORE INTO discoveries (
                    document_id, search_query, search_provider,
                    referring_url, discovery_timestamp
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    candidate.query,
                    candidate.provider,
                    candidate.referring_url or "",
                    discovered,
                ),
            )
            return document_id, is_new

    def add_local_document(
        self,
        *,
        source_url: str,
        canonical_url: str,
        title: str,
        local_path: str,
        file_size: int,
        sha256: str,
        duplicate_of: int | None,
    ) -> tuple[int, bool]:
        discovered = utc_now()
        status = "duplicate_content" if duplicate_of is not None else "imported"
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT id FROM documents WHERE canonical_url = ?",
                (canonical_url,),
            ).fetchone()
            if existing is not None:
                return int(existing["id"]), False
            cursor = connection.execute(
                """
                INSERT INTO documents (
                    canonical_url, source_url, title, discovery_timestamp,
                    download_timestamp, local_path, file_size, sha256,
                    download_status, duplicate_of_document_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    canonical_url,
                    source_url,
                    title,
                    discovered,
                    discovered,
                    local_path,
                    file_size,
                    sha256,
                    status,
                    duplicate_of,
                    discovered,
                ),
            )
            document_id = int(cursor.lastrowid)
            connection.execute(
                """
                INSERT INTO discoveries (
                    document_id, search_query, search_provider,
                    discovery_timestamp
                ) VALUES (?, 'manual local import', 'manual', ?)
                """,
                (document_id, discovered),
            )
            return document_id, True

    def queue_candidates(self, candidates: list[Candidate]) -> tuple[int, int]:
        new_count = 0
        duplicate_count = 0
        for candidate in candidates:
            _, is_new = self.add_candidate(candidate)
            new_count += int(is_new)
            duplicate_count += int(not is_new)
        return new_count, duplicate_count

    def record_api_request(self, provider: str, query: str) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO api_requests (
                    provider, search_query, requested_at, status
                ) VALUES (?, ?, ?, 'started')
                """,
                (provider, query, utc_now()),
            )
            return int(cursor.lastrowid)

    def finish_api_request(
        self,
        request_id: int,
        status: str,
        result_count: int = 0,
        error_message: str | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE api_requests
                SET completed_at = ?, status = ?, result_count = ?, error_message = ?
                WHERE id = ?
                """,
                (utc_now(), status, result_count, error_message, request_id),
            )

    def request_count(self) -> int:
        with self.connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM api_requests").fetchone()
            return int(row["count"])

    def get_document(self, document_id: int) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()

    def pending_documents(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(
                connection.execute(
                    """
                    SELECT * FROM documents
                    WHERE download_status IN ('discovered', 'failed')
                    ORDER BY id
                    """
                )
            )

    def mark_download_failed(self, document_id: int, message: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE documents
                SET download_status = 'failed', error_message = ?
                WHERE id = ?
                """,
                (message[:1000], document_id),
            )

    def find_hash(self, sha256: str, exclude_id: int) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT * FROM documents
                WHERE sha256 = ? AND id != ?
                  AND download_status IN ('downloaded', 'duplicate_content', 'imported')
                ORDER BY id LIMIT 1
                """,
                (sha256, exclude_id),
            ).fetchone()

    def save_download(self, document_id: int, result: DownloadResult) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE documents
                SET download_status = ?, local_path = ?, file_size = ?, sha256 = ?,
                    download_timestamp = ?, error_message = NULL,
                    duplicate_of_document_id = ?
                WHERE id = ?
                """,
                (
                    result.status,
                    result.local_path,
                    result.file_size,
                    result.sha256,
                    utc_now(),
                    result.duplicate_of,
                    document_id,
                ),
            )

    def statistics(self) -> dict[str, int]:
        with self.connect() as connection:
            counts = {
                row["download_status"]: int(row["count"])
                for row in connection.execute(
                    "SELECT download_status, COUNT(*) AS count FROM documents GROUP BY download_status"
                )
            }
            counts["documents"] = sum(counts.values())
            counts["api_requests"] = int(
                connection.execute("SELECT COUNT(*) FROM api_requests").fetchone()[0]
            )
            return counts

    def list_documents(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(connection.execute("SELECT * FROM documents ORDER BY id"))

    def discoveries_for(self, document_id: int) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(
                connection.execute(
                    "SELECT * FROM discoveries WHERE document_id = ? ORDER BY id",
                    (document_id,),
                )
            )

    def api_request_rows(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(connection.execute("SELECT * FROM api_requests ORDER BY id"))
