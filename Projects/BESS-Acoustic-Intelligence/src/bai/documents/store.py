from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from bai.database import Database
from bai.documents.classification import CLASSIFIER_VERSION
from bai.documents.models import DocumentClassification, PDFParseResult


logger = logging.getLogger(__name__)
SCHEMA_VERSION = 3
BACKUP_SUFFIX = "-phase2-v1-backup"
PROCESSABLE_STATUSES = {"downloaded", "duplicate_content", "imported"}


class DocumentStore:
    def __init__(self, database: Database):
        self.database = database

    def initialize(self) -> None:
        path = self.database.path
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path, timeout=30) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            current_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if current_version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"Database schema version {current_version} is newer than supported "
                    f"version {SCHEMA_VERSION}"
                )
            if current_version < SCHEMA_VERSION:
                self._backup_before_b1_migration(connection, path)
                self._backup_legacy_database(connection, path)
                if current_version < 1:
                    self._run_migration(connection, _MIGRATION_SQL)
                    current_version = 1
                if current_version < 2:
                    self._run_migration(connection, _MIGRATION_V2_SQL)
                if current_version < 3:
                    self._run_migration(connection, _MIGRATION_V3_SQL)
                logger.info(
                    "Initialized document intelligence database schema version %s",
                    SCHEMA_VERSION,
                )

    @staticmethod
    def _run_migration(connection: sqlite3.Connection, migration_sql: str) -> None:
        try:
            connection.executescript(migration_sql)
        except sqlite3.Error:
            connection.rollback()
            raise

    @staticmethod
    def _backup_legacy_database(connection: sqlite3.Connection, path: Path) -> None:
        has_documents = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='documents'"
        ).fetchone()
        if not has_documents:
            return
        backup_path = path.with_name(path.name + BACKUP_SUFFIX)
        if backup_path.exists():
            return
        with sqlite3.connect(backup_path) as backup:
            connection.backup(backup)
        logger.info("Created pre-migration SQLite backup at %s", backup_path)

    @staticmethod
    def _backup_before_b1_migration(
        connection: sqlite3.Connection,
        path: Path,
    ) -> None:
        has_documents = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='documents'"
        ).fetchone()
        if not has_documents:
            return
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_path = path.with_name(
            f"{path.name}-phase2b1-{timestamp}-backup"
        )
        with sqlite3.connect(backup_path) as backup:
            connection.backup(backup)
        with sqlite3.connect(backup_path) as backup:
            integrity = backup.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(
                    f"Pre-migration backup failed integrity check: {backup_path}"
                )
        logger.info("Created and verified pre-B1 SQLite backup at %s", backup_path)

    def set_provider_eligibility(
        self,
        document_id: int,
        eligible: bool,
        reviewer: str,
        authorization_basis: str,
    ) -> None:
        if not reviewer.strip():
            raise ValueError("A reviewer is required to change provider eligibility")
        if not authorization_basis.strip():
            raise ValueError("An authorization basis is required to change provider eligibility")
        with self.database.connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if exists is None:
                raise ValueError(f"Unknown document ID {document_id}")
            connection.execute(
                """
                INSERT INTO provider_processing_eligibility_events (
                    document_id, eligible, reviewer, authorization_basis, recorded_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    int(eligible),
                    reviewer.strip(),
                    authorization_basis.strip(),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def provider_eligibility(self) -> list[sqlite3.Row]:
        with self.database.connect() as connection:
            return list(
                connection.execute(
                    """
                    SELECT d.id AS document_id, d.title, d.source_url,
                           COALESCE(e.eligible, 0) AS eligible,
                           e.reviewer, e.authorization_basis, e.recorded_at
                    FROM documents AS d
                    LEFT JOIN provider_processing_eligibility_events AS e
                      ON e.id = (
                          SELECT MAX(latest.id)
                          FROM provider_processing_eligibility_events AS latest
                          WHERE latest.document_id = d.id
                      )
                    ORDER BY d.id
                    """
                )
            )

    def is_provider_processing_eligible(self, document_id: int) -> bool:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT eligible
                FROM provider_processing_eligibility_events
                WHERE document_id = ?
                ORDER BY id DESC LIMIT 1
                """,
                (document_id,),
            ).fetchone()
        return bool(row["eligible"]) if row is not None else False

    def processable_documents(self, limit: int | None = None) -> list[sqlite3.Row]:
        documents = [
            row
            for row in self.database.list_documents()
            if row["download_status"] in PROCESSABLE_STATUSES and row["local_path"]
        ]
        return documents if limit is None else documents[:limit]

    def has_parse(self, document_id: int, sha256: str, parser_version: str) -> bool:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT status FROM document_processing
                WHERE document_id = ? AND source_sha256 = ? AND parser_version = ?
                """,
                (document_id, sha256, parser_version),
            ).fetchone()
        return row is not None and row["status"] in {"complete", "partial"}

    def save_parse(self, document_id: int, result: PDFParseResult) -> int:
        digest = result.source_sha256 or ""
        metadata_json = _json(result.metadata)
        warnings_json = _json(
            [
                {
                    "code": warning.code,
                    "message": warning.message,
                    "page_number": warning.page_number,
                    "review_flag": warning.review_flag,
                }
                for warning in result.warnings
            ]
        )
        flags_json = _json(result.review_flags)
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT id, status FROM document_processing
                WHERE document_id = ? AND source_sha256 = ? AND parser_version = ?
                """,
                (document_id, digest, result.parser_version),
            ).fetchone()
            if row is None:
                cursor = connection.execute(
                    """
                    INSERT INTO document_processing (
                        document_id, source_sha256, parser_version, status,
                        processed_at, page_count, metadata_json, warnings_json,
                        review_flags_json
                    ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        digest,
                        result.parser_version,
                        result.status,
                        len(result.pages),
                        metadata_json,
                        warnings_json,
                        flags_json,
                    ),
                )
                processing_id = int(cursor.lastrowid)
            else:
                processing_id = int(row["id"])
                if row["status"] in {"complete", "partial"}:
                    return processing_id
                connection.execute(
                    """
                    UPDATE document_processing
                    SET status = ?, processed_at = CURRENT_TIMESTAMP,
                        page_count = ?, metadata_json = ?, warnings_json = ?,
                        review_flags_json = ?
                    WHERE id = ?
                    """,
                    (
                        result.status,
                        len(result.pages),
                        metadata_json,
                        warnings_json,
                        flags_json,
                        processing_id,
                    ),
                )
                connection.execute(
                    "DELETE FROM extracted_pages WHERE processing_id = ?",
                    (processing_id,),
                )
                connection.execute(
                    "DELETE FROM extracted_tables WHERE processing_id = ?",
                    (processing_id,),
                )
                connection.execute(
                    "DELETE FROM source_evidence WHERE processing_id = ?",
                    (processing_id,),
                )

            for page in result.pages:
                connection.execute(
                    """
                    INSERT INTO extracted_pages (
                        processing_id, document_id, page_number, text, text_status,
                        text_character_count, image_count, acoustic_signals_json,
                        review_flags_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        processing_id,
                        document_id,
                        page.page_number,
                        page.text,
                        page.text_status,
                        page.text_character_count,
                        page.image_count,
                        _json(page.acoustic_signals),
                        _json(page.review_flags),
                    ),
                )
                for signal in page.acoustic_signals:
                    connection.execute(
                        """
                        INSERT INTO source_evidence (
                            processing_id, document_id, page_number, evidence_type,
                            evidence_text, evidence_status, extraction_method,
                            review_status
                        ) VALUES (?, ?, ?, 'acoustic_signal', ?, 'present',
                                  'deterministic_text_match', 'unreviewed')
                        """,
                        (processing_id, document_id, page.page_number, signal),
                    )

            for table in result.tables:
                cursor = connection.execute(
                    """
                    INSERT INTO extracted_tables (
                        processing_id, document_id, page_number, table_index,
                        row_count, column_count, cells_json, column_headers_json,
                        header_row_index, heading_context,
                        frequency_band_labels_json, units_json, bounding_box_json,
                        logical_row_count, logical_column_count, logical_cells_json,
                        header_row_indices_json, frequency_bands_json,
                        measurement_types_json, measurement_type_evidence_json,
                        alignment_status, review_flags_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        processing_id,
                        document_id,
                        table.page_number,
                        table.table_index,
                        table.row_count,
                        table.column_count,
                        _json(
                            [
                                {
                                    "row_index": cell.row_index,
                                    "column_index": cell.column_index,
                                    "text": cell.text,
                                    "bounding_box": cell.bounding_box,
                                }
                                for cell in table.cells
                            ]
                        ),
                        _json(table.column_headers),
                        table.header_row_index,
                        table.heading_context,
                        _json(table.frequency_band_labels),
                        _json(table.units),
                        _json(table.bounding_box) if table.bounding_box is not None else None,
                        table.logical_row_count,
                        table.logical_column_count,
                        _json(
                            [
                                {
                                    "row_index": cell.row_index,
                                    "column_start": cell.column_start,
                                    "column_end": cell.column_end,
                                    "text": cell.text,
                                    "source_row_index": cell.source_row_index,
                                    "source_column_index": cell.source_column_index,
                                    "bounding_box": cell.bounding_box,
                                    "row_start": cell.row_start,
                                    "row_end": cell.row_end,
                                }
                                for cell in table.logical_cells
                            ]
                        ),
                        _json(table.header_row_indices),
                        _json(
                            [
                                {
                                    "source_label": band.source_label,
                                    "center_hz": band.center_hz,
                                    "logical_column": band.logical_column,
                                    "source_row_index": band.source_row_index,
                                }
                                for band in table.frequency_bands
                            ]
                        ),
                        _json(table.measurement_types),
                        _json(table.measurement_type_evidence),
                        table.alignment_status,
                        _json(table.review_flags),
                    ),
                )
                table_id = int(cursor.lastrowid)
                for cell in table.cells:
                    if cell.text:
                        connection.execute(
                            """
                            INSERT INTO source_evidence (
                                processing_id, document_id, page_number, table_id,
                                row_index, column_index, evidence_type, evidence_text,
                                evidence_status, extraction_method, review_status
                            ) VALUES (?, ?, ?, ?, ?, ?, 'table_cell', ?, 'present',
                                      'pymupdf_table', 'unreviewed')
                            """,
                            (
                                processing_id,
                                document_id,
                                table.page_number,
                                table_id,
                                cell.row_index,
                                cell.column_index,
                                cell.text,
                            ),
                        )
            return processing_id

    def classification_tasks(self, limit: int | None = None) -> list[sqlite3.Row]:
        query = """
            SELECT dp.id AS processing_id, dp.document_id, dp.source_sha256,
                   dp.review_flags_json, d.title, d.source_url
            FROM document_processing AS dp
            JOIN documents AS d ON d.id = dp.document_id
            WHERE dp.status IN ('complete', 'partial', 'failed')
              AND dp.source_sha256 = COALESCE(d.sha256, '')
            ORDER BY dp.document_id, dp.id DESC
        """
        with self.database.connect() as connection:
            rows = list(connection.execute(query))
            seen: set[int] = set()
            latest = []
            for row in rows:
                document_id = int(row["document_id"])
                if document_id not in seen:
                    latest.append(row)
                    seen.add(document_id)
        return latest if limit is None else latest[:limit]

    def pages_for_processing(self, processing_id: int) -> list[sqlite3.Row]:
        with self.database.connect() as connection:
            return list(
                connection.execute(
                    """
                    SELECT page_number, text, text_status, text_character_count,
                           image_count, acoustic_signals_json, review_flags_json
                    FROM extracted_pages
                    WHERE processing_id = ? ORDER BY page_number
                    """,
                    (processing_id,),
                )
            )

    def has_classification(self, processing_id: int, classifier_version: str) -> bool:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM document_classifications
                WHERE processing_id = ? AND classifier_version = ? LIMIT 1
                """,
                (processing_id, classifier_version),
            ).fetchone()
        return row is not None

    def save_classifications(
        self,
        processing_id: int,
        document_id: int,
        classifications: tuple[DocumentClassification, ...],
        classifier_version: str,
    ) -> None:
        with self.database.connect() as connection:
            connection.execute(
                "DELETE FROM document_classifications WHERE processing_id = ? AND classifier_version = ?",
                (processing_id, classifier_version),
            )
            for classification in classifications:
                cursor = connection.execute(
                    """
                    INSERT INTO document_classifications (
                        processing_id, document_id, classifier_version, category,
                        score, direct_bess_evidence, cross_industry, is_primary,
                        review_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        processing_id,
                        document_id,
                        classifier_version,
                        classification.category,
                        classification.score,
                        int(classification.direct_bess_evidence),
                        int(classification.cross_industry),
                        int(classification.is_primary),
                        classification.review_status,
                    ),
                )
                classification_id = int(cursor.lastrowid)
                for item in classification.evidence:
                    connection.execute(
                        """
                        INSERT INTO source_evidence (
                            processing_id, document_id, classification_id,
                            page_number, evidence_type, evidence_text,
                            evidence_status, extraction_method, review_status
                        ) VALUES (?, ?, ?, ?, 'classification', ?, ?,
                                  'deterministic_keyword_match', ?)
                        """,
                        (
                            processing_id,
                            document_id,
                            classification_id,
                            item.page_number,
                            f"{item.matched_term}: {item.excerpt}",
                            item.evidence_status,
                            classification.review_status,
                        ),
                    )

    def statistics(self) -> dict[str, int]:
        with self.database.connect() as connection:
            totals = {
                "processing_runs": int(
                    connection.execute("SELECT COUNT(*) FROM document_processing").fetchone()[0]
                ),
                "parsed_pages": int(
                    connection.execute("SELECT COUNT(*) FROM extracted_pages").fetchone()[0]
                ),
                "candidate_tables": int(
                    connection.execute("SELECT COUNT(*) FROM extracted_tables").fetchone()[0]
                ),
                "classifications": int(
                    connection.execute("SELECT COUNT(*) FROM document_classifications").fetchone()[0]
                ),
                "source_evidence": int(
                    connection.execute("SELECT COUNT(*) FROM source_evidence").fetchone()[0]
                ),
            }
            latest = list(
                connection.execute(
                    """
                    SELECT dp.id, dp.document_id, dp.status, dp.review_flags_json
                    FROM document_processing AS dp
                    JOIN (
                        SELECT document_id, MAX(id) AS id
                        FROM document_processing
                        GROUP BY document_id
                    ) AS current ON current.id = dp.id
                    """
                )
            )
            latest_ids = [int(row["id"]) for row in latest]
            processing_documents = {
                int(row["id"]): int(row["document_id"]) for row in latest
            }
            page_rows = []
            table_rows = []
            classification_rows = []
            if latest_ids:
                placeholders = ",".join("?" for _ in latest_ids)
                page_rows = list(
                    connection.execute(
                        f"""
                        SELECT processing_id, text_status, review_flags_json
                        FROM extracted_pages
                        WHERE processing_id IN ({placeholders})
                        """,
                        latest_ids,
                    )
                )
                table_rows = list(
                    connection.execute(
                        f"""
                        SELECT processing_id, alignment_status, review_flags_json,
                               measurement_types_json
                        FROM extracted_tables
                        WHERE processing_id IN ({placeholders})
                        """,
                        latest_ids,
                    )
                )
                classification_rows = list(
                    connection.execute(
                        f"""
                        SELECT document_id, review_status
                        FROM document_classifications
                        WHERE processing_id IN ({placeholders})
                          AND classifier_version = ?
                        """,
                        [*latest_ids, CLASSIFIER_VERSION],
                    )
                )

        review_documents: set[int] = set()
        outstanding_review_items = 0
        parse_failures = 0
        for row in latest:
            flags = json.loads(row["review_flags_json"])
            outstanding_review_items += len(flags)
            if flags:
                review_documents.add(int(row["document_id"]))
            parse_failures += int(row["status"] == "failed")
        for row in page_rows:
            flags = json.loads(row["review_flags_json"])
            outstanding_review_items += len(flags)
            if flags:
                review_documents.add(processing_documents[int(row["processing_id"])])
        for row in table_rows:
            flags = json.loads(row["review_flags_json"])
            outstanding_review_items += len(flags)
            if flags:
                review_documents.add(processing_documents[int(row["processing_id"])])
        for row in classification_rows:
            if row["review_status"] == "needs_review":
                outstanding_review_items += 1
                review_documents.add(int(row["document_id"]))

        totals.update(
            {
                "latest_processed_documents": len(latest),
                "latest_parsed_pages": len(page_rows),
                "latest_candidate_tables": len(table_rows),
                "successfully_extracted_tables": sum(
                    row["alignment_status"] in {"aligned", "reconstructed"}
                    for row in table_rows
                ),
                "missing_text_pages": sum(
                    row["text_status"] != "extracted" for row in page_rows
                ),
                "parse_failures": parse_failures,
                "ambiguous_tables": sum(
                    row["alignment_status"] == "ambiguous" for row in table_rows
                ),
                "missing_measurement_type_tables": sum(
                    "unspecified" in json.loads(row["measurement_types_json"])
                    for row in table_rows
                ),
                "outstanding_review_items": outstanding_review_items,
                "review_required_documents": len(review_documents),
            }
        )
        return totals


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


_MIGRATION_SQL = """
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS document_processing (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source_sha256 TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('complete', 'partial', 'failed')),
    processed_at TEXT NOT NULL,
    page_count INTEGER NOT NULL DEFAULT 0 CHECK (page_count >= 0),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    warnings_json TEXT NOT NULL DEFAULT '[]',
    review_flags_json TEXT NOT NULL DEFAULT '[]',
    UNIQUE (document_id, source_sha256, parser_version)
);
CREATE TABLE IF NOT EXISTS extracted_pages (
    id INTEGER PRIMARY KEY,
    processing_id INTEGER NOT NULL REFERENCES document_processing(id) ON DELETE CASCADE,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL CHECK (page_number > 0),
    text TEXT NOT NULL,
    text_status TEXT NOT NULL CHECK (
        text_status IN ('extracted', 'image_heavy', 'no_extractable_text', 'extraction_failed')
    ),
    text_character_count INTEGER NOT NULL CHECK (text_character_count >= 0),
    image_count INTEGER NOT NULL DEFAULT 0 CHECK (image_count >= 0),
    acoustic_signals_json TEXT NOT NULL DEFAULT '[]',
    review_flags_json TEXT NOT NULL DEFAULT '[]',
    UNIQUE (processing_id, page_number)
);
CREATE TABLE IF NOT EXISTS extracted_tables (
    id INTEGER PRIMARY KEY,
    processing_id INTEGER NOT NULL REFERENCES document_processing(id) ON DELETE CASCADE,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL CHECK (page_number > 0),
    table_index INTEGER NOT NULL CHECK (table_index >= 0),
    row_count INTEGER NOT NULL CHECK (row_count >= 0),
    column_count INTEGER NOT NULL CHECK (column_count >= 0),
    cells_json TEXT NOT NULL,
    column_headers_json TEXT NOT NULL DEFAULT '[]',
    header_row_index INTEGER,
    heading_context TEXT NOT NULL DEFAULT '',
    frequency_band_labels_json TEXT NOT NULL DEFAULT '[]',
    units_json TEXT NOT NULL DEFAULT '[]',
    bounding_box_json TEXT,
    UNIQUE (processing_id, page_number, table_index)
);
CREATE TABLE IF NOT EXISTS document_classifications (
    id INTEGER PRIMARY KEY,
    processing_id INTEGER NOT NULL REFERENCES document_processing(id) ON DELETE CASCADE,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    classifier_version TEXT NOT NULL,
    category TEXT NOT NULL,
    score REAL NOT NULL CHECK (score >= 0 AND score <= 1),
    direct_bess_evidence INTEGER NOT NULL CHECK (direct_bess_evidence IN (0, 1)),
    cross_industry INTEGER NOT NULL CHECK (cross_industry IN (0, 1)),
    is_primary INTEGER NOT NULL CHECK (is_primary IN (0, 1)),
    review_status TEXT NOT NULL CHECK (review_status IN ('unreviewed', 'needs_review')),
    UNIQUE (processing_id, classifier_version, category)
);
CREATE TABLE IF NOT EXISTS source_evidence (
    id INTEGER PRIMARY KEY,
    processing_id INTEGER NOT NULL REFERENCES document_processing(id) ON DELETE CASCADE,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    classification_id INTEGER REFERENCES document_classifications(id) ON DELETE CASCADE,
    table_id INTEGER REFERENCES extracted_tables(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL CHECK (page_number > 0),
    row_index INTEGER,
    column_index INTEGER,
    evidence_type TEXT NOT NULL,
    evidence_text TEXT NOT NULL,
    evidence_status TEXT NOT NULL CHECK (
        evidence_status IN ('present', 'not_found', 'not_extractable', 'ambiguous', 'conflicting')
    ),
    extraction_method TEXT NOT NULL,
    review_status TEXT NOT NULL CHECK (
        review_status IN ('unreviewed', 'needs_review', 'reviewed')
    ),
    CHECK (row_index IS NULL OR row_index >= 0),
    CHECK (column_index IS NULL OR column_index >= 0)
);
CREATE TABLE IF NOT EXISTS equipment (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source_evidence_id INTEGER NOT NULL REFERENCES source_evidence(id),
    equipment_category TEXT,
    source_type TEXT,
    manufacturer TEXT,
    model TEXT,
    unit_count INTEGER CHECK (unit_count IS NULL OR unit_count > 0),
    operating_conditions TEXT,
    availability_status TEXT NOT NULL DEFAULT 'present' CHECK (
        availability_status IN ('present', 'not_found', 'not_extractable', 'ambiguous', 'conflicting')
    ),
    review_status TEXT NOT NULL DEFAULT 'needs_review' CHECK (
        review_status IN ('unreviewed', 'needs_review', 'reviewed')
    )
);
CREATE TABLE IF NOT EXISTS acoustic_measurements (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source_evidence_id INTEGER NOT NULL REFERENCES source_evidence(id),
    numeric_level REAL,
    units TEXT,
    metric_type TEXT CHECK (metric_type IS NULL OR metric_type IN ('sound_power', 'sound_pressure')),
    weighting TEXT,
    measurement_distance REAL,
    distance_units TEXT,
    frequency_band_center REAL,
    frequency_bandwidth REAL,
    frequency_units TEXT,
    tonal_frequency REAL,
    operating_load TEXT,
    fan_speed TEXT,
    value_basis TEXT CHECK (value_basis IS NULL OR value_basis IN ('measured', 'predicted', 'manufacturer_supplied', 'unknown')),
    aggregation_basis TEXT CHECK (aggregation_basis IS NULL OR aggregation_basis IN ('per_unit', 'aggregate_facility', 'unknown')),
    uncertainty REAL,
    confidence REAL CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    availability_status TEXT NOT NULL DEFAULT 'present' CHECK (
        availability_status IN ('present', 'not_found', 'not_extractable', 'ambiguous', 'conflicting')
    ),
    review_status TEXT NOT NULL DEFAULT 'needs_review' CHECK (
        review_status IN ('unreviewed', 'needs_review', 'reviewed')
    ),
    CHECK (numeric_level IS NULL OR (units IS NOT NULL AND metric_type IS NOT NULL)),
    CHECK (measurement_distance IS NULL OR distance_units IS NOT NULL)
);
CREATE TABLE IF NOT EXISTS mitigation_strategies (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source_evidence_id INTEGER NOT NULL REFERENCES source_evidence(id),
    mitigation_type TEXT,
    equipment_targeted TEXT,
    reported_attenuation REAL,
    attenuation_units TEXT,
    frequency_specific_performance TEXT,
    cooling_or_airflow_penalty TEXT,
    installation_footprint TEXT,
    reported_cost REAL,
    cost_units TEXT,
    deployment_evidence TEXT,
    availability_status TEXT NOT NULL DEFAULT 'present' CHECK (
        availability_status IN ('present', 'not_found', 'not_extractable', 'ambiguous', 'conflicting')
    ),
    review_status TEXT NOT NULL DEFAULT 'needs_review' CHECK (
        review_status IN ('unreviewed', 'needs_review', 'reviewed')
    ),
    CHECK (reported_attenuation IS NULL OR attenuation_units IS NOT NULL),
    CHECK (reported_cost IS NULL OR cost_units IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS idx_processing_document ON document_processing(document_id);
CREATE INDEX IF NOT EXISTS idx_processing_digest ON document_processing(source_sha256);
CREATE INDEX IF NOT EXISTS idx_pages_document ON extracted_pages(document_id, page_number);
CREATE INDEX IF NOT EXISTS idx_tables_document ON extracted_tables(document_id, page_number);
CREATE INDEX IF NOT EXISTS idx_classifications_category ON document_classifications(category, score);
CREATE INDEX IF NOT EXISTS idx_evidence_document ON source_evidence(document_id, page_number);
PRAGMA user_version = 1;
COMMIT;
"""

_MIGRATION_V2_SQL = """
BEGIN IMMEDIATE;
ALTER TABLE extracted_tables ADD COLUMN logical_row_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE extracted_tables ADD COLUMN logical_column_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE extracted_tables ADD COLUMN logical_cells_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE extracted_tables ADD COLUMN header_row_indices_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE extracted_tables ADD COLUMN frequency_bands_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE extracted_tables ADD COLUMN measurement_types_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE extracted_tables ADD COLUMN measurement_type_evidence_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE extracted_tables ADD COLUMN alignment_status TEXT NOT NULL DEFAULT 'ambiguous';
ALTER TABLE extracted_tables ADD COLUMN review_flags_json TEXT NOT NULL DEFAULT '[]';
CREATE INDEX IF NOT EXISTS idx_tables_alignment ON extracted_tables(alignment_status);
PRAGMA user_version = 2;
COMMIT;
"""

_MIGRATION_V3_SQL = """
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS provider_processing_eligibility_events (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    eligible INTEGER NOT NULL CHECK (eligible IN (0, 1)),
    reviewer TEXT NOT NULL CHECK (length(trim(reviewer)) > 0),
    authorization_basis TEXT NOT NULL CHECK (length(trim(authorization_basis)) > 0),
    recorded_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_provider_eligibility_document
    ON provider_processing_eligibility_events(document_id, id);
PRAGMA user_version = 3;
COMMIT;
"""
