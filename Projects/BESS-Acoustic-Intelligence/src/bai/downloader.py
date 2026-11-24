import hashlib
import logging
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

import requests

from bai.database import Database
from bai.models import DownloadResult
from bai.web import PoliteWebClient


logger = logging.getLogger(__name__)
CHUNK_SIZE = 64 * 1024


class PDFDownloader:
    def __init__(
        self,
        database: Database,
        web: PoliteWebClient,
        download_dir: str | Path,
        *,
        max_retries: int,
    ):
        self.database = database
        self.web = web
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max(1, max_retries)

    def download(self, document_id: int) -> DownloadResult:
        document = self.database.get_document(document_id)
        if document is None:
            raise ValueError(f"Unknown document id: {document_id}")
        if document["download_status"] in {"downloaded", "duplicate_content", "imported"}:
            return DownloadResult(
                status=document["download_status"],
                local_path=document["local_path"],
                file_size=document["file_size"],
                sha256=document["sha256"],
                duplicate_of=document["duplicate_of_document_id"],
            )

        filename = f"{document_id}-{_safe_stem(document['title'], document['source_url'])}.pdf"
        destination = self.download_dir / filename
        partial = destination.with_suffix(destination.suffix + ".part")
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                self._download_attempt(document["source_url"], partial)
                size = partial.stat().st_size
                if not _is_pdf(partial):
                    partial.unlink(missing_ok=True)
                    raise ValueError(f"Response from {document['source_url']} is not a valid PDF")
                digest = _sha256(partial)
                existing = self.database.find_hash(digest, document_id)
                if existing is not None:
                    partial.unlink(missing_ok=True)
                    result = DownloadResult(
                        status="duplicate_content",
                        local_path=existing["local_path"],
                        file_size=size,
                        sha256=digest,
                        duplicate_of=int(existing["id"]),
                    )
                else:
                    partial.replace(destination)
                    result = DownloadResult(
                        status="downloaded",
                        local_path=str(destination),
                        file_size=size,
                        sha256=digest,
                    )
                self.database.save_download(document_id, result)
                return result
            except PermissionError as exc:
                last_error = exc
                break
            except ValueError as exc:
                last_error = exc
                break
            except (requests.RequestException, OSError) as exc:
                last_error = exc
                logger.warning(
                    "Download attempt %s/%s failed for %s: %s",
                    attempt + 1,
                    self.max_retries,
                    document["source_url"],
                    exc,
                )
                if attempt + 1 < self.max_retries:
                    delay = _retry_delay(exc, attempt)
                    if delay is None:
                        logger.warning(
                            "Server requested a long retry delay; leaving %s queued",
                            document["source_url"],
                        )
                        break
                    time.sleep(delay)

        message = str(last_error or "Download failed")
        self.database.mark_download_failed(document_id, message)
        raise RuntimeError(f"Could not download {document['source_url']}: {message}") from last_error

    def _download_attempt(self, url: str, partial: Path) -> None:
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        response = self.web.get(url, headers=headers)
        try:
            if response.status_code == 416 and offset:
                partial.unlink(missing_ok=True)
                raise requests.RequestException("Server rejected the partial download range")
            response.raise_for_status()
            append = offset > 0 and response.status_code == 206
            if append:
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {offset}-"):
                    partial.unlink(missing_ok=True)
                    raise requests.RequestException(
                        "Server returned an invalid Content-Range for the resumed download"
                    )
            mode = "ab" if append else "wb"
            with partial.open(mode) as output:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        output.write(chunk)
        finally:
            response.close()


def _safe_stem(title: str, url: str) -> str:
    candidate = title.strip() or Path(unquote(urlsplit(url).path)).stem or "document"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate).strip("-.")
    return (stem or "document")[:100]


def _is_pdf(path: Path) -> bool:
    with path.open("rb") as file:
        header = file.read(1024)
        file.seek(max(0, path.stat().st_size - 1024))
        trailer = file.read()
    return b"%PDF-" in header and b"%%EOF" in trailer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _retry_delay(error: Exception, attempt: int) -> float | None:
    response = error.response if isinstance(error, requests.HTTPError) else None
    retry_after = response.headers.get("Retry-After") if response is not None else None
    if retry_after:
        try:
            delay = float(retry_after)
        except ValueError:
            try:
                retry_time = parsedate_to_datetime(retry_after)
            except (TypeError, ValueError, OverflowError):
                delay = min(2**attempt, 8)
            else:
                if retry_time.tzinfo is None:
                    retry_time = retry_time.replace(tzinfo=timezone.utc)
                delay = max(0.0, (retry_time - datetime.now(timezone.utc)).total_seconds())
        if delay > 60:
            return None
        return max(delay, 0.0)
    return min(2**attempt, 8)
