from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import replace
from pathlib import Path

from bai.collection import resolve_project_path
from bai.database import Database
from bai.documents.classification import CLASSIFIER_VERSION, classify_pages
from bai.documents.models import (
    ExtractedPage,
    PDFParseResult,
    ParseWarning,
    ProcessingSummary,
)
from bai.documents.parser import PARSER_VERSION, PDFParser
from bai.documents.store import DocumentStore


logger = logging.getLogger(__name__)


class DocumentIntelligence:
    def __init__(
        self,
        database: Database,
        *,
        parser: PDFParser | None = None,
    ):
        self.database = database
        self.store = DocumentStore(database)
        self.parser = parser or PDFParser()
        self.store.initialize()

    def parse_documents(
        self,
        *,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> ProcessingSummary:
        if limit is not None and limit < 0:
            raise ValueError("limit must be zero or greater")
        documents = self.store.processable_documents()
        pending: list[tuple[object, Path, str, PDFParseResult | None]] = []
        skipped = 0
        for document in documents:
            path = _resolve_document_path(document["local_path"])
            try:
                digest = _sha256(path)
            except OSError:
                digest = None
            expected_digest = document["sha256"]
            if digest is not None and expected_digest and digest != expected_digest:
                pending.append(
                    (
                        document,
                        path,
                        digest,
                        _hash_mismatch_result(path, digest),
                    )
                )
                continue
            effective_digest = digest or expected_digest or ""
            if self.store.has_parse(int(document["id"]), effective_digest, PARSER_VERSION):
                skipped += 1
                continue
            pending.append((document, path, effective_digest, None))

        if limit is not None:
            pending = pending[:limit]
        if dry_run:
            return ProcessingSummary(
                processed=len(pending),
                skipped=skipped,
                outstanding_review=self.store.statistics()["outstanding_review_items"],
            )

        processed = 0
        failed = 0
        review_required = 0
        for document, path, expected_digest, precomputed_result in pending:
            result = precomputed_result or self.parser.parse(path)
            if result.source_sha256 is None and expected_digest:
                result = replace(result, source_sha256=expected_digest)
            self.store.save_parse(int(document["id"]), result)
            if "source_hash_mismatch" in result.review_flags:
                logger.warning(
                    "Document %s has a local file hash that differs from its stored SHA-256",
                    document["id"],
                )
            processed += 1
            failed += int(result.status == "failed")
            review_required += int(bool(result.review_flags))
        return ProcessingSummary(
            processed=processed,
            skipped=skipped,
            failed=failed,
            review_required=review_required,
            outstanding_review=self.store.statistics()["outstanding_review_items"],
        )

    def classify_documents(
        self,
        *,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> ProcessingSummary:
        if limit is not None and limit < 0:
            raise ValueError("limit must be zero or greater")
        tasks = self.store.classification_tasks()
        eligible = [
            row
            for row in tasks
            if not self.store.has_classification(
                int(row["processing_id"]),
                CLASSIFIER_VERSION,
            )
        ]
        skipped = len(tasks) - len(eligible)
        pending = eligible if limit is None else eligible[:limit]
        if dry_run:
            return ProcessingSummary(
                processed=len(pending),
                skipped=skipped,
                outstanding_review=self.store.statistics()["outstanding_review_items"],
            )

        processed = 0
        review_required = 0
        for row in pending:
            page_rows = self.store.pages_for_processing(int(row["processing_id"]))
            pages = tuple(
                ExtractedPage(
                    page_number=int(page["page_number"]),
                    text=page["text"],
                    text_status=page["text_status"],
                    text_character_count=int(page["text_character_count"]),
                    image_count=int(page["image_count"]),
                    acoustic_signals=tuple(json.loads(page["acoustic_signals_json"])),
                    review_flags=tuple(json.loads(page["review_flags_json"])),
                )
                for page in page_rows
            )
            flags = tuple(json.loads(row["review_flags_json"]))
            classifications = classify_pages(
                pages,
                flags,
                document_title=row["title"] or "",
            )
            self.store.save_classifications(
                int(row["processing_id"]),
                int(row["document_id"]),
                classifications,
                CLASSIFIER_VERSION,
            )
            processed += 1
            review_required += int(
                any(item.review_status == "needs_review" for item in classifications)
            )
        return ProcessingSummary(
            processed=processed,
            skipped=skipped,
            review_required=review_required,
            outstanding_review=self.store.statistics()["outstanding_review_items"],
        )


def _resolve_document_path(local_path: str) -> Path:
    path = Path(local_path).expanduser()
    return path if path.is_absolute() else resolve_project_path(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_mismatch_result(path: Path, actual_digest: str) -> PDFParseResult:
    return PDFParseResult(
        source_path=path,
        source_sha256=actual_digest,
        parser_version=PARSER_VERSION,
        status="failed",
        metadata={},
        pages=(),
        tables=(),
        warnings=(
            ParseWarning(
                "stored_hash_mismatch",
                "The local file does not match the SHA-256 stored for this document; parsing was skipped.",
                review_flag="manual_review",
            ),
        ),
        review_flags=("manual_review", "source_hash_mismatch"),
    )
