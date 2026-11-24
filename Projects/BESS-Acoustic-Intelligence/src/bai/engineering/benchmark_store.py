from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from bai.engineering.schemas import (
    AcousticMeasurement,
    BenchmarkCandidate,
    BenchmarkReviewItem,
    GoldStandardRecord,
    HumanReviewedEvidence,
    ReviewEvent,
)


class BenchmarkStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=30) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > 2:
                raise RuntimeError(
                    f"Benchmark database version {version} is newer than supported version 2"
                )
            has_candidates = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='benchmark_candidates'"
            ).fetchone()
            if has_candidates and version < 2:
                columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(benchmark_candidates)"
                    )
                }
                if version == 1 or "identity_key" not in columns:
                    self._backup_before_migration(connection)
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS benchmark_candidates (
                    candidate_id TEXT PRIMARY KEY,
                    identity_key TEXT NOT NULL UNIQUE,
                    generator_version TEXT NOT NULL DEFAULT 'stage-b1.0',
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('pending', 'rejected', 'approved')
                    ),
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS benchmark_review_events (
                    id INTEGER PRIMARY KEY,
                    candidate_id TEXT NOT NULL
                        REFERENCES benchmark_candidates(candidate_id),
                    action TEXT NOT NULL CHECK (
                        action IN ('corrected', 'rejected', 'approved')
                    ),
                    reviewer TEXT NOT NULL CHECK (length(trim(reviewer)) > 0),
                    reason TEXT NOT NULL DEFAULT '',
                    event_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS gold_standard_records (
                    candidate_id TEXT PRIMARY KEY
                        REFERENCES benchmark_candidates(candidate_id),
                    record_json TEXT NOT NULL,
                    approved_by TEXT NOT NULL,
                    approved_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS benchmark_review_items (
                    review_item_id TEXT PRIMARY KEY,
                    generator_version TEXT NOT NULL,
                    source_key TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('open', 'resolved')),
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_benchmark_status
                    ON benchmark_candidates(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_benchmark_review_status
                    ON benchmark_review_items(status, generator_version);
                CREATE INDEX IF NOT EXISTS idx_benchmark_review_candidate
                    ON benchmark_review_events(candidate_id, id);
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(benchmark_candidates)")
            }
            if "identity_key" not in columns:
                if version >= 1:
                    raise RuntimeError(
                        "Benchmark database schema version 1 is missing identity_key"
                    )
                connection.execute(
                    "ALTER TABLE benchmark_candidates ADD COLUMN identity_key TEXT"
                )
                rows = list(
                    connection.execute(
                        "SELECT candidate_id, payload_json FROM benchmark_candidates"
                    )
                )
                for candidate_id, payload_json in rows:
                    connection.execute(
                        "UPDATE benchmark_candidates SET identity_key = ? "
                        "WHERE candidate_id = ?",
                        (_identity_key(json.loads(payload_json), "stage-b1.0"), candidate_id),
                    )
            if "generator_version" not in columns:
                connection.execute(
                    "ALTER TABLE benchmark_candidates ADD COLUMN generator_version "
                    "TEXT NOT NULL DEFAULT 'stage-b1.0'"
                )
            if version < 2:
                rows = list(
                    connection.execute(
                        "SELECT candidate_id, identity_key, payload_json, generator_version "
                        "FROM benchmark_candidates"
                    )
                )
                for candidate_id, _, payload_json, generator_version in rows:
                    version_name = generator_version or "stage-b1.0"
                    connection.execute(
                        "UPDATE benchmark_candidates SET identity_key = ?, "
                        "generator_version = ? WHERE candidate_id = ?",
                        (
                            _identity_key(json.loads(payload_json), version_name),
                            version_name,
                            candidate_id,
                        ),
                    )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_benchmark_generator_version "
                "ON benchmark_candidates(generator_version, status)"
            )
            connection.execute("PRAGMA user_version = 2")

    def _backup_before_migration(self, source: sqlite3.Connection) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_path = self.path.with_name(
            f"{self.path.stem}-schema-v2-{timestamp}{self.path.suffix}"
        )
        counts = {
            table: source.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "benchmark_candidates",
                "benchmark_review_events",
                "gold_standard_records",
            )
            if source.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
        }
        with sqlite3.connect(backup_path) as backup:
            source.backup(backup)
        with sqlite3.connect(backup_path) as backup:
            integrity = backup.execute("PRAGMA integrity_check").fetchone()[0]
            foreign_key_errors = backup.execute("PRAGMA foreign_key_check").fetchall()
            copied_counts = {
                table: backup.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in counts
            }
            if integrity != "ok" or foreign_key_errors or counts != copied_counts:
                raise RuntimeError(
                    f"Benchmark database backup failed integrity check: {backup_path}"
                )

    def add_candidates(self, candidates: list[BenchmarkCandidate]) -> tuple[int, int]:
        added = 0
        existing = 0
        with sqlite3.connect(self.path, timeout=30) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            for candidate in candidates:
                payload = candidate.model_dump_json()
                generator_version = candidate.generator_version
                row = connection.execute(
                    "SELECT payload_json, generator_version FROM benchmark_candidates "
                    "WHERE candidate_id = ?",
                    (candidate.candidate_id,),
                ).fetchone()
                if row is not None:
                    if (
                        row[1] != generator_version
                        or _candidate_semantics(json.loads(row[0]))
                        != _candidate_semantics(json.loads(payload))
                    ):
                        raise RuntimeError(
                            f"Candidate ID collision with different source data: "
                            f"{candidate.candidate_id}"
                        )
                    existing += 1
                    continue
                identity_key = _identity_key(json.loads(payload), generator_version)
                semantic_match = connection.execute(
                    """
                    SELECT payload_json FROM benchmark_candidates
                    WHERE identity_key = ?
                    """,
                    (identity_key,),
                ).fetchone()
                if semantic_match is not None:
                    if _candidate_semantics(json.loads(semantic_match[0])) != _candidate_semantics(
                        json.loads(payload)
                    ):
                        raise RuntimeError(
                            f"Candidate source identity collision: {candidate.candidate_id}"
                        )
                    existing += 1
                    continue
                connection.execute(
                    """
                    INSERT INTO benchmark_candidates (
                        candidate_id, identity_key, generator_version, payload_json,
                        status, created_at
                    ) VALUES (?, ?, ?, ?, 'pending', ?)
                    """,
                    (
                        candidate.candidate_id,
                        identity_key,
                        generator_version,
                        payload,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                added += 1
        return added, existing

    def add_review_items(self, items: list[BenchmarkReviewItem]) -> tuple[int, int]:
        added = 0
        existing = 0
        with sqlite3.connect(self.path, timeout=30) as connection:
            for item in items:
                payload = item.model_dump_json()
                source_key = _review_source_key(item)
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO benchmark_review_items (
                        review_item_id, generator_version, source_key, payload_json,
                        status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.review_item_id,
                        item.generator_version,
                        source_key,
                        payload,
                        item.status,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                if cursor.rowcount:
                    added += 1
                else:
                    existing += 1
        return added, existing

    def review_items(self, status: str | None = "open") -> list[BenchmarkReviewItem]:
        query = "SELECT payload_json FROM benchmark_review_items"
        parameters: tuple[str, ...] = ()
        if status is not None:
            if status not in {"open", "resolved"}:
                raise ValueError(f"Unsupported review item status: {status}")
            query += " WHERE status = ?"
            parameters = (status,)
        query += " ORDER BY generator_version, created_at, review_item_id"
        with sqlite3.connect(self.path) as connection:
            rows = list(connection.execute(query, parameters))
        return [BenchmarkReviewItem.model_validate_json(row[0]) for row in rows]

    def get_candidate(self, candidate_id: str) -> BenchmarkCandidate:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM benchmark_candidates
                WHERE candidate_id = ?
                """,
                (candidate_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"Unknown benchmark candidate: {candidate_id}")
        return BenchmarkCandidate.model_validate_json(row[0])

    def list_candidates(
        self,
        status: str | None = None,
        generator_version: str | None = None,
    ) -> list[tuple[str, str, BenchmarkCandidate]]:
        query = """
            SELECT candidate_id, status, payload_json
            FROM benchmark_candidates
        """
        clauses = []
        parameters: list[str] = []
        if status is not None:
            if status not in {"pending", "rejected", "approved"}:
                raise ValueError(f"Unsupported candidate status: {status}")
            clauses.append("status = ?")
            parameters.append(status)
        if generator_version is not None:
            if generator_version not in {"stage-b1.0", "stage-b1.1"}:
                raise ValueError(f"Unsupported candidate generation version: {generator_version}")
            clauses.append("generator_version = ?")
            parameters.append(generator_version)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY candidate_id"
        with sqlite3.connect(self.path) as connection:
            rows = list(connection.execute(query, parameters))
        return [
            (row[0], row[1], BenchmarkCandidate.model_validate_json(row[2]))
            for row in rows
        ]

    def current_measurement(self, candidate_id: str) -> AcousticMeasurement:
        return self.current_review(candidate_id)[0]

    def current_review(
        self,
        candidate_id: str,
    ) -> tuple[AcousticMeasurement, tuple[HumanReviewedEvidence, ...]]:
        candidate = self.get_candidate(candidate_id)
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                """
                SELECT event_json FROM benchmark_review_events
                WHERE candidate_id = ? AND action = 'corrected'
                ORDER BY id DESC LIMIT 1
                """,
                (candidate_id,),
            ).fetchone()
        if row is None:
            return candidate.measurement, ()
        event = ReviewEvent.model_validate_json(row[0])
        assert event.corrected_measurement is not None
        return event.corrected_measurement, event.human_review_evidence

    def record_correction(
        self,
        candidate_id: str,
        corrected: AcousticMeasurement,
        reviewer: str,
        reason: str,
        human_evidence: tuple[HumanReviewedEvidence, ...],
    ) -> None:
        event = ReviewEvent(
            candidate_id=candidate_id,
            action="corrected",
            reviewer=reviewer,
            reason=reason,
            recorded_at=datetime.now(timezone.utc),
            corrected_measurement=corrected,
            human_review_evidence=human_evidence,
        )
        self._record_event(event, "pending")

    def reject(self, candidate_id: str, reviewer: str, reason: str) -> None:
        event = ReviewEvent(
            candidate_id=candidate_id,
            action="rejected",
            reviewer=reviewer,
            reason=reason,
            recorded_at=datetime.now(timezone.utc),
        )
        self._record_event(event, "rejected")

    def approve(
        self,
        candidate_id: str,
        reviewer: str,
        reason: str,
        measurement: AcousticMeasurement,
        human_evidence: tuple[HumanReviewedEvidence, ...] = (),
    ) -> GoldStandardRecord:
        candidate = self.get_candidate(candidate_id)
        approved_at = datetime.now(timezone.utc)
        verified_measurement = AcousticMeasurement.model_validate(
            {
                **measurement.model_dump(),
                "transcription_status": "human_verified",
            }
        )
        record = GoldStandardRecord(
            candidate_id=candidate_id,
            source=candidate.source,
            generator_version=candidate.generator_version,
            page_number=candidate.page_number,
            table_id=candidate.table_id,
            source_table_reference=candidate.source_table_reference,
            record_kind=candidate.record_kind,
            bess_relevance=candidate.bess_relevance,
            source_entity_role=candidate.source_entity_role,
            source_entity_label=candidate.source_entity_label,
            equipment=candidate.equipment,
            measurement=verified_measurement,
            evidence=candidate.evidence,
            page_context=candidate.page_context,
            human_review_evidence=human_evidence,
            approved_by=reviewer,
            approved_at=approved_at,
        )
        event = ReviewEvent(
            candidate_id=candidate_id,
            action="approved",
            reviewer=reviewer,
            reason=reason,
            recorded_at=approved_at,
        )
        with sqlite3.connect(self.path, timeout=30) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            self._require_pending(connection, candidate_id)
            connection.execute(
                """
                INSERT INTO benchmark_review_events (
                    candidate_id, action, reviewer, reason, event_json, recorded_at
                ) VALUES (?, 'approved', ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    reviewer,
                    reason,
                    event.model_dump_json(),
                    approved_at.isoformat(),
                ),
            )
            connection.execute(
                """
                INSERT INTO gold_standard_records (
                    candidate_id, record_json, approved_by, approved_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    record.model_dump_json(),
                    reviewer,
                    approved_at.isoformat(),
                ),
            )
            connection.execute(
                "UPDATE benchmark_candidates SET status = 'approved' WHERE candidate_id = ?",
                (candidate_id,),
            )
        return record

    def _record_event(self, event: ReviewEvent, new_status: str) -> None:
        if not event.reviewer.strip():
            raise ValueError("A reviewer name is required")
        with sqlite3.connect(self.path, timeout=30) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            self._require_pending(connection, event.candidate_id)
            connection.execute(
                """
                INSERT INTO benchmark_review_events (
                    candidate_id, action, reviewer, reason, event_json, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.candidate_id,
                    event.action,
                    event.reviewer,
                    event.reason,
                    event.model_dump_json(),
                    event.recorded_at.isoformat(),
                ),
            )
            if new_status != "pending":
                connection.execute(
                    "UPDATE benchmark_candidates SET status = ? WHERE candidate_id = ?",
                    (new_status, event.candidate_id),
                )

    @staticmethod
    def _require_pending(connection: sqlite3.Connection, candidate_id: str) -> None:
        row = connection.execute(
            "SELECT status FROM benchmark_candidates WHERE candidate_id = ?",
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown benchmark candidate: {candidate_id}")
        if row[0] != "pending":
            raise ValueError(
                f"Candidate {candidate_id} is {row[0]}; only pending candidates may be reviewed"
            )

    def review_history(self, candidate_id: str) -> list[ReviewEvent]:
        with sqlite3.connect(self.path) as connection:
            rows = list(
                connection.execute(
                    """
                    SELECT event_json FROM benchmark_review_events
                    WHERE candidate_id = ? ORDER BY id
                    """,
                    (candidate_id,),
                )
            )
        return [ReviewEvent.model_validate_json(row[0]) for row in rows]

    def gold_records(self) -> list[GoldStandardRecord]:
        with sqlite3.connect(self.path) as connection:
            rows = list(
                connection.execute(
                    "SELECT record_json FROM gold_standard_records ORDER BY approved_at"
                )
            )
        return [GoldStandardRecord.model_validate_json(row[0]) for row in rows]

    def statistics(self) -> dict[str, int]:
        with sqlite3.connect(self.path) as connection:
            counts = {
                row[0]: int(row[1])
                for row in connection.execute(
                    "SELECT status, COUNT(*) FROM benchmark_candidates GROUP BY status"
                )
            }
            counts["review_events"] = int(
                connection.execute(
                    "SELECT COUNT(*) FROM benchmark_review_events"
                ).fetchone()[0]
            )
            counts["gold_records"] = int(
                connection.execute(
                    "SELECT COUNT(*) FROM gold_standard_records"
                ).fetchone()[0]
            )
            counts["open_review_items"] = int(
                connection.execute(
                    "SELECT COUNT(*) FROM benchmark_review_items WHERE status = 'open'"
                ).fetchone()[0]
            )
            counts["review_items"] = int(
                connection.execute(
                    "SELECT COUNT(*) FROM benchmark_review_items"
                ).fetchone()[0]
            )
            counts["stage-b1.0"] = int(
                connection.execute(
                    "SELECT COUNT(*) FROM benchmark_candidates "
                    "WHERE generator_version = 'stage-b1.0'"
                ).fetchone()[0]
            )
            counts["stage-b1.1"] = int(
                connection.execute(
                    "SELECT COUNT(*) FROM benchmark_candidates "
                    "WHERE generator_version = 'stage-b1.1'"
                ).fetchone()[0]
            )
        for status in ("pending", "rejected", "approved"):
            counts.setdefault(status, 0)
        return counts


def _candidate_semantics(payload: dict) -> str:
    normalized = BenchmarkCandidate.model_validate(payload).model_dump(mode="json")
    normalized.pop("candidate_id", None)
    normalized.pop("table_id", None)
    for citation in normalized.get("evidence", []):
        citation.pop("evidence_id", None)
        citation.pop("table_id", None)
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False)


def _identity_key(payload: dict, generator_version: str) -> str:
    semantic = _candidate_semantics(payload)
    return hashlib.sha256(f"{generator_version}\0{semantic}".encode("utf-8")).hexdigest()


def _review_source_key(item: BenchmarkReviewItem) -> str:
    identity = {
        "version": item.generator_version,
        "sha256": item.source.sha256,
        "page": item.page_number,
        "table_index": item.table_index,
        "row_index": item.row_index,
        "column_index": item.column_index,
        "reason": item.reason,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
