from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import defaultdict, deque
from decimal import Decimal
from pathlib import Path

from bai.database import Database
from bai.engineering.benchmark_store import BenchmarkStore
from bai.engineering.schemas import (
    AcousticMeasurement,
    BenchmarkCandidate,
    DocumentSource,
    EquipmentInformation,
    FrequencyBandInformation,
    SourceEvidenceReference,
)
from bai.engineering.validation import (
    EvidenceValidationError,
    parse_unambiguous_number,
    validate_candidate_evidence,
)


_ACOUSTIC_CONTEXT = re.compile(
    r"\bdB(?:\s*\(|[A-Z]|\b)|sound\s+(?:power|pressure)|"
    r"insertion\s+loss|sound\s+reduction\s+index|\bSRI\b|"
    r"\b(?:LAeq|LwA|LpA|Leq)\b",
    re.I,
)
_UNIT = re.compile(r"dB\s*(?:\([^)]{1,12}\)|re\s+[^\n;,]{1,40}|[A-Z])?", re.I)
_METRIC_PATTERNS = (
    ("sound_power", re.compile(r"\bsound\s+power\b|\bpower\s+level\b", re.I)),
    ("sound_pressure", re.compile(r"\bsound\s+pressure\b|\bpressure\s+level\b|\bSPL\b", re.I)),
    ("sound_reduction_index", re.compile(r"\bsound\s+reduction\s+index\b|\bSRI\b", re.I)),
    ("insertion_loss", re.compile(r"\binsertion\s+loss\b", re.I)),
)


class BenchmarkGenerator:
    def __init__(self, database: Database, benchmark_store: BenchmarkStore):
        self.database = database
        self.benchmark_store = benchmark_store

    def generate(self, limit: int = 50) -> tuple[int, int, int]:
        if not 1 <= limit <= 50:
            raise ValueError("Candidate generation limit must be between 1 and 50")
        self.benchmark_store.initialize()
        candidates_by_document = self._candidate_groups()
        selected = _round_robin(candidates_by_document, limit)
        added, existing = self.benchmark_store.add_candidates(selected)
        return added, existing, len(selected)

    def _candidate_groups(self) -> dict[int, list[BenchmarkCandidate]]:
        with self.database.connect() as connection:
            tables = list(
                connection.execute(
                    """
                    SELECT et.*, d.title, d.source_url, d.local_path, d.sha256,
                           dp.source_sha256
                    FROM extracted_tables AS et
                    JOIN document_processing AS dp ON dp.id = et.processing_id
                    JOIN documents AS d ON d.id = et.document_id
                    WHERE et.processing_id = (
                        SELECT latest.id FROM document_processing AS latest
                        WHERE latest.document_id = d.id
                          AND latest.source_sha256 = COALESCE(d.sha256, '')
                        ORDER BY latest.id DESC LIMIT 1
                    )
                      AND dp.status IN ('complete', 'partial')
                      AND d.download_status IN ('downloaded', 'duplicate_content', 'imported')
                      AND et.alignment_status IN ('aligned', 'reconstructed')
                      AND d.sha256 IS NOT NULL AND d.local_path IS NOT NULL
                    ORDER BY d.id, et.page_number, et.table_index
                    """
                )
            )
            evidence_rows = list(
                connection.execute(
                    """
                    SELECT id, document_id, page_number, table_id, row_index,
                           column_index, evidence_text, evidence_status
                    FROM source_evidence
                    WHERE table_id IS NOT NULL AND evidence_type = 'table_cell'
                    ORDER BY id
                    """
                )
            )
        evidence_by_table: dict[int, dict[tuple[int, int], sqlite3.Row]] = defaultdict(dict)
        for row in evidence_rows:
            if row["row_index"] is not None and row["column_index"] is not None:
                evidence_by_table[int(row["table_id"])][
                    (int(row["row_index"]), int(row["column_index"]))
                ] = row

        grouped: dict[int, list[BenchmarkCandidate]] = defaultdict(list)
        for table in tables:
            table_id = int(table["id"])
            document_id = int(table["document_id"])
            raw_evidence = evidence_by_table.get(table_id, {})
            cells = json.loads(table["logical_cells_json"])
            header_rows = set(json.loads(table["header_row_indices_json"]))
            bands = json.loads(table["frequency_bands_json"])
            horizontal_bands = {
                int(band["logical_column"]): band
                for band in bands
                if band.get("logical_column") is not None
                and int(band["source_row_index"]) in header_rows
            }
            vertical_bands = {
                int(band["source_row_index"]): band
                for band in bands
                if band.get("logical_column") is not None
                and int(band["source_row_index"]) not in header_rows
            }
            candidates = self._table_candidates(
                table=table,
                cells=cells,
                header_rows=header_rows,
                horizontal_bands=horizontal_bands,
                vertical_bands=vertical_bands,
                raw_evidence=raw_evidence,
            )
            for candidate in candidates:
                validate_candidate_evidence(candidate, self.database)
                grouped[document_id].append(candidate)
        return grouped

    def _table_candidates(
        self,
        *,
        table,
        cells: list[dict],
        header_rows: set[int],
        horizontal_bands: dict[int, dict],
        vertical_bands: dict[int, dict],
        raw_evidence: dict[tuple[int, int], object],
    ) -> list[BenchmarkCandidate]:
        table_id = int(table["id"])
        document_id = int(table["document_id"])
        source = DocumentSource(
            document_id=document_id,
            title=table["title"] or "",
            filename=Path(table["local_path"]).name,
            original_url=table["source_url"],
            sha256=table["sha256"],
        )
        max_header_row = max(header_rows, default=-1)
        candidates: list[BenchmarkCandidate] = []
        seen: set[tuple[int, int, str]] = set()

        def consider(cell: dict, band: dict | None) -> None:
            text = cell["text"].strip()
            number = parse_unambiguous_number(text)
            if not text or number is None:
                return
            row_index = int(cell["row_index"])
            column_start = cell.get("column_start")
            column_end = cell.get("column_end")
            if column_start is None or column_end is None or column_start != column_end:
                return
            source_row = int(cell["source_row_index"])
            source_column = int(cell["source_column_index"])
            value_row = raw_evidence.get((source_row, source_column))
            if value_row is None or value_row["evidence_status"] != "present":
                return
            unique_key = (source_row, source_column, text)
            if unique_key in seen:
                return
            seen.add(unique_key)

            headers = [
                header
                for header in cells
                if int(header["row_index"]) in header_rows
                and header.get("column_start") is not None
                and int(header["column_start"]) <= int(column_start)
                <= int(header.get("column_end", header["column_start"]))
                and header.get("text", "").strip()
            ]
            header_text = " ".join(header["text"] for header in headers)
            if band is None and not _ACOUSTIC_CONTEXT.search(header_text):
                return
            if (
                band is not None
                and int(band["source_row_index"]) > max_header_row
                and not _ACOUSTIC_CONTEXT.search(header_text)
            ):
                return

            citations: list[SourceEvidenceReference] = []

            def add_citation(raw_cell: dict, role: str) -> None:
                evidence = raw_evidence.get(
                    (
                        int(raw_cell["source_row_index"]),
                        int(raw_cell["source_column_index"]),
                    )
                )
                if evidence is None or evidence["evidence_status"] != "present":
                    return
                citation = SourceEvidenceReference(
                    evidence_id=int(evidence["id"]),
                    role=role,
                    document_id=document_id,
                    page_number=int(evidence["page_number"]),
                    table_id=table_id,
                    row_index=int(evidence["row_index"]),
                    column_index=int(evidence["column_index"]),
                    quote=evidence["evidence_text"],
                )
                if not any(
                    item.evidence_id == citation.evidence_id and item.role == citation.role
                    for item in citations
                ):
                    citations.append(citation)

            add_citation(cell, "measurement_value")
            metric_type = "unknown"
            for header in headers:
                header_value = header["text"].strip()
                for name, matcher in _METRIC_PATTERNS:
                    if matcher.search(header_value):
                        metric_type = name
                        add_citation(header, "metric_context")
                if re.search(r"\b(?:measured|measurement)\b", header_value, re.I):
                    add_citation(header, "basis")
                elif re.search(r"\bpredict(?:ed|ion)\b|\bmodeled\b|\bmodelled\b", header_value, re.I):
                    add_citation(header, "basis")
                elif re.search(r"\bmanufacturer[- ]supplied\b", header_value, re.I):
                    add_citation(header, "basis")

            unit_text = _unit_text(text)
            unit_source = value_row["evidence_text"] if unit_text else None
            if not unit_text:
                for header in headers:
                    unit_text = _unit_text(header["text"])
                    if unit_text:
                        unit_source = header["text"]
                        add_citation(header, "unit")
                        break
            if unit_text and unit_source == value_row["evidence_text"]:
                add_citation(cell, "unit")
            weighting = _weighting(unit_text or "")
            if weighting != "unknown":
                for citation in tuple(citations):
                    if citation.role == "unit":
                        citations.append(
                            citation.model_copy(update={"role": "weighting"})
                        )

            frequency = None
            if band is not None:
                frequency_cell = next(
                    (
                        item
                        for item in cells
                        if int(item["row_index"]) == int(band["source_row_index"])
                        and item.get("column_start") is not None
                        and int(item["column_start"])
                        <= int(band["logical_column"])
                        <= int(item.get("column_end", item["column_start"]))
                        and item.get("text", "").strip()
                    ),
                    None,
                )
                if frequency_cell is None:
                    return
                add_citation(frequency_cell, "frequency_band")
                source_bandwidth_text = next(
                    (
                        header["text"].strip()
                        for header in headers
                        if re.search(r"one[- ]third octave|octave band|octave", header["text"], re.I)
                    ),
                    None,
                )
                if source_bandwidth_text:
                    bandwidth_cell = next(
                        header
                        for header in headers
                        if header["text"].strip() == source_bandwidth_text
                    )
                    add_citation(bandwidth_cell, "frequency_band")
                frequency = FrequencyBandInformation(
                    center_hz=Decimal(str(band["center_hz"])),
                    source_label=band["source_label"],
                    bandwidth=_bandwidth(source_bandwidth_text),
                    source_bandwidth_text=source_bandwidth_text,
                )

            row_label = _row_label(cells, cell, int(column_start))
            equipment = EquipmentInformation(source_label=row_label)
            if row_label:
                label_cell = _row_label_cell(cells, cell, int(column_start))
                if label_cell is not None:
                    add_citation(label_cell, "equipment_label")

            source_value_text = value_row["evidence_text"]
            try:
                measurement = AcousticMeasurement(
                    value=number,
                    source_value_text=source_value_text,
                    units=unit_text,
                    source_unit_text=unit_source,
                    metric_type=metric_type,
                    weighting=weighting,
                    frequency_band=frequency,
                )
                identity = {
                    "document_id": document_id,
                    "source_sha256": source.sha256,
                    "page_number": int(table["page_number"]),
                    "table_index": int(table["table_index"]),
                    "source_row": source_row,
                    "source_column": source_column,
                    "equipment": equipment.model_dump(mode="json"),
                    "measurement": measurement.model_dump(mode="json"),
                    "evidence": [
                        {
                            "role": item.role,
                            "page_number": item.page_number,
                            "row_index": item.row_index,
                            "column_index": item.column_index,
                            "quote": item.quote,
                        }
                        for item in citations
                    ],
                }
                material = json.dumps(identity, sort_keys=True, ensure_ascii=False)
                candidate_id = (
                    f"cand-{hashlib.sha256(material.encode('utf-8')).hexdigest()[:16]}"
                )
                candidate = BenchmarkCandidate(
                    candidate_id=candidate_id,
                    source=source,
                    page_number=int(table["page_number"]),
                    table_id=table_id,
                    equipment=equipment,
                    measurement=measurement,
                    evidence=tuple(citations),
                )
                candidates.append(candidate)
            except ValueError as exc:
                raise EvidenceValidationError(
                    f"Could not construct candidate from table {table_id}, "
                    f"row {row_index}, column {column_start}: {exc}"
                ) from exc

        for cell in cells:
            row_index = int(cell["row_index"])
            if row_index <= max_header_row or cell.get("row_start", row_index) != row_index:
                continue
            column_start = cell.get("column_start")
            column_end = cell.get("column_end")
            if column_start is None or column_end is None:
                continue
            row_bands = vertical_bands.get(row_index)
            if (
                row_bands is not None
                and int(column_start) == int(row_bands["logical_column"])
                and int(column_end) == int(row_bands["logical_column"])
            ):
                continue
            if column_start == column_end and int(column_start) in horizontal_bands:
                consider(cell, horizontal_bands[int(column_start)])
                continue
            if any(
                column_start <= column <= column_end for column in horizontal_bands
            ):
                continue
            if row_bands is not None:
                consider(cell, row_bands)
            else:
                consider(cell, None)
        return candidates


def _round_robin(
    grouped: dict[int, list[BenchmarkCandidate]],
    limit: int,
) -> list[BenchmarkCandidate]:
    queues = [deque(grouped[key]) for key in sorted(grouped)]
    selected = []
    while queues and len(selected) < limit:
        remaining = []
        for queue in queues:
            if queue and len(selected) < limit:
                selected.append(queue.popleft())
            if queue:
                remaining.append(queue)
        queues = remaining
    return selected


def _unit_text(text: str) -> str | None:
    match = _UNIT.search(text)
    if not match:
        return None
    return re.sub(r"\s+", " ", match.group(0)).strip()


def _weighting(unit: str) -> str:
    if re.search(r"dB\s*\(\s*A\s*\)|dBA", unit, re.I):
        return "A"
    if re.search(r"dB\s*\(\s*C\s*\)|dBC", unit, re.I):
        return "C"
    if re.search(r"dB\s*\(\s*Z\s*\)|dBZ", unit, re.I):
        return "Z"
    if re.search(r"\bunweighted\b", unit, re.I):
        return "unweighted"
    return "unknown"


def _bandwidth(text: str | None) -> str:
    if not text:
        return "unknown"
    if re.search(r"one[- ]third octave", text, re.I):
        return "one_third_octave"
    if re.search(r"octave", text, re.I):
        return "octave"
    if re.search(r"narrowband", text, re.I):
        return "narrowband"
    return "unknown"


def _row_label(cells: list[dict], value_cell: dict, value_column: int) -> str | None:
    label_cell = _row_label_cell(cells, value_cell, value_column)
    return label_cell["text"].strip() if label_cell else None


def _row_label_cell(
    cells: list[dict],
    value_cell: dict,
    value_column: int,
) -> dict | None:
    row_index = int(value_cell["row_index"])
    labels = [
        cell
        for cell in cells
        if int(cell["row_index"]) == row_index
        and cell.get("column_end") is not None
        and int(cell["column_end"]) < value_column
        and cell.get("text", "").strip()
        and parse_unambiguous_number(cell["text"]) is None
    ]
    return min(labels, key=lambda cell: int(cell["column_start"])) if labels else None
