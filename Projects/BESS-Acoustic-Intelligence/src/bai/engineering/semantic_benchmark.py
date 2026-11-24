from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from bai.database import Database
from bai.engineering.benchmark import _round_robin
from bai.engineering.benchmark_store import BenchmarkStore
from bai.engineering.schemas import (
    AcousticMeasurement,
    BenchmarkCandidate,
    BenchmarkReviewItem,
    DocumentSource,
    EquipmentInformation,
    FrequencyBandInformation,
    PageContextEvidence,
    SourceEvidenceReference,
)
from bai.engineering.validation import (
    EvidenceValidationError,
    validate_candidate_evidence,
)


GENERATOR_VERSION = "stage-b1.1"
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_EXACT = re.compile(rf"^\s*({_NUMBER})\s*(.*?)\s*$")
_BOUND = re.compile(rf"^\s*([<>≤≥]=?)\s*({_NUMBER})\s*(.*?)\s*$")
_RANGE = re.compile(
    rf"^\s*({_NUMBER})\s*(?:-|–|—|to)\s*({_NUMBER})\s*(.*?)\s*$",
    re.I,
)
_FREQUENCY = re.compile(
    rf"^\s*({_NUMBER})\s*(kHz|Hz|kilohertz|hertz|k)?\s*$", re.I
)
_UNIT = re.compile(
    r"\b(?:decibels?|dB)\s*(?:\(\s*[ABCZ]\s*\)|[ABCZ]\b|re\s*"
    r"(?:10\s*(?:\^?\s*[-−]?12)\s*(?:W|watts?)|"
    r"20\s*(?:µ|μ|u)?\s*Pa))?",
    re.I,
)
_ACOUSTIC = re.compile(
    r"\bdB\b|sound|noise|acoustic|octave|frequency|SPL|SRI|"
    r"\bL\s*[_\s]?(?:Aeq|AF90|A90|AFmax|Amax|den|night|day)\b",
    re.I,
)
_METRIC_PATTERNS = (
    ("sound_power", re.compile(r"\bsound\s+power\b|\bSWL\b|\bL\s*W\s*[A-Z]?\b", re.I)),
    ("sound_pressure", re.compile(r"\bsound\s+pressure\b|\bSPL\b|dB\s*re\s*20\s*(?:µ|μ|u)?\s*Pa", re.I)),
    ("sound_reduction_index", re.compile(r"\bsound\s+reduction\s+index\b|\bSRI\b", re.I)),
    ("insertion_loss", re.compile(r"\binsertion\s+loss\b", re.I)),
)
_FREQUENCY_HEADER = re.compile(
    r"\bfrequency\b|\bfreq\.?\b|octave[- ]band(?:\s+(?:centre|center))?",
    re.I,
)
_EXCLUDED_COLUMN = re.compile(
    r"\b(?:quantity|qty|number\s+of\s+sources?|distance|time|period|"
    r"monitoring\s+location|receiver|receptor|margin|compliance|"
    r"easting|northing|coordinates?|date|duration|serial\s+number|"
    r"frequency|freq\.?|centre\s+frequency|center\s+frequency)\b",
    re.I,
)
_EQUIPMENT_WORDS = re.compile(
    r"\b(?:inverter|transformer|battery(?:\s+container|\s+unit)?|PCS|PCU|"
    r"HVAC|fan|cooling\s+system|chiller|container|switchgear|"
    r"power\s+conversion\s+(?:system|unit))\b",
    re.I,
)
_GENERIC_REFERENCE = re.compile(
    r"\b(?:jet\s+aircraft|jet\s+flyover|wilderness|typical\s+sound|"
    r"common\s+sound\s+sources?|reference\s+(?:sound|noise)\s+levels?)\b",
    re.I,
)
_BESS = re.compile(
    r"\bBESS\b|battery\s+(?:energy\s+)?storage"
    r"(?:\s+(?:system|equipment|units?|containers?|enclosures?))?|"
    r"\benergy\s+storage\s+system\b",
    re.I,
)
_BESS_LINKED_CONTEXT = re.compile(
    r"\bBESS\s+(?:equipment|systems?|facilities|projects?|containers?|units?|"
    r"inverters?|transformers?|site)\b|"
    r"\bbattery\s+(?:energy\s+)?storage\s+"
    r"(?:equipment|systems?|facilities|projects?|containers?|units?|enclosures?)\b",
    re.I,
)
_TABLE_REFERENCE = re.compile(r"\btable\s+\d+(?:[-–]\d+)?\b", re.I)


@dataclass(frozen=True)
class NumericInterpretation:
    comparator: str
    value: Decimal | None = None
    lower_bound: Decimal | None = None
    upper_bound: Decimal | None = None


class SemanticBenchmarkGenerator:
    def __init__(self, database: Database, benchmark_store: BenchmarkStore):
        self.database = database
        self.benchmark_store = benchmark_store
        self.last_report: dict[str, int] = {}

    def generate(self, limit: int = 1000) -> tuple[int, int, int]:
        if not 1 <= limit <= 5000:
            raise ValueError("B1.1 candidate generation limit must be between 1 and 5000")
        self.benchmark_store.initialize()
        candidates_by_document, review_items = self._candidate_groups()
        selected = _round_robin(candidates_by_document, limit)
        added, existing = self.benchmark_store.add_candidates(selected)
        review_added, review_existing = self.benchmark_store.add_review_items(review_items)
        self.last_report = {
            "selected": len(selected),
            "added": added,
            "existing": existing,
            "review_items_added": review_added,
            "review_items_existing": review_existing,
        }
        return added, existing, len(selected)

    def collect_candidates(
        self,
    ) -> tuple[dict[int, list[BenchmarkCandidate]], list[BenchmarkReviewItem]]:
        return self._candidate_groups()

    def _candidate_groups(
        self,
    ) -> tuple[dict[int, list[BenchmarkCandidate]], list[BenchmarkReviewItem]]:
        with self.database.connect() as connection:
            tables = list(
                connection.execute(
                    """
                    SELECT et.*, d.title, d.source_url, d.local_path, d.sha256,
                           dp.source_sha256, ep.text AS page_text
                    FROM extracted_tables AS et
                    JOIN document_processing AS dp ON dp.id = et.processing_id
                    JOIN documents AS d ON d.id = et.document_id
                    JOIN extracted_pages AS ep
                      ON ep.processing_id = et.processing_id
                     AND ep.page_number = et.page_number
                    WHERE et.processing_id = (
                        SELECT latest.id FROM document_processing AS latest
                        WHERE latest.document_id = d.id
                          AND latest.source_sha256 = COALESCE(d.sha256, '')
                        ORDER BY latest.id DESC LIMIT 1
                    )
                      AND dp.status IN ('complete', 'partial')
                      AND d.download_status IN ('downloaded', 'duplicate_content', 'imported')
                      AND d.sha256 IS NOT NULL AND d.local_path IS NOT NULL
                      AND d.id = (
                          SELECT MIN(d2.id) FROM documents AS d2
                          WHERE d2.sha256 = d.sha256
                            AND d2.download_status IN (
                                'downloaded', 'duplicate_content', 'imported'
                            )
                      )
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
        review_items: list[BenchmarkReviewItem] = []
        for table in tables:
            table_id = int(table["id"])
            document_id = int(table["document_id"])
            cells = json.loads(table["logical_cells_json"])
            headers = set(json.loads(table["header_row_indices_json"]))
            source_evidence = evidence_by_table.get(table_id, {})
            source = DocumentSource(
                document_id=document_id,
                title=table["title"] or "",
                filename=Path(table["local_path"]).name,
                original_url=table["source_url"],
                sha256=table["sha256"],
            )
            table_ref = _table_reference(
                table["heading_context"], table["page_text"], cells
            )
            if table["alignment_status"] == "ambiguous":
                review_items.append(
                    _review_item(
                        source,
                        table,
                        table_ref,
                        "ambiguous_table_alignment",
                        table["heading_context"],
                    )
                )
                continue
            if table["alignment_status"] not in {"aligned", "reconstructed"}:
                review_items.append(
                    _review_item(
                        source,
                        table,
                        table_ref,
                        "table_alignment_not_eligible",
                        table["heading_context"],
                    )
                )
                continue

            candidates, table_reviews = self._table_candidates(
                table=table,
                source=source,
                cells=cells,
                header_rows=headers,
                raw_evidence=source_evidence,
                source_table_reference=table_ref,
            )
            review_items.extend(table_reviews)
            for candidate in candidates:
                try:
                    validate_candidate_evidence(candidate, self.database)
                except EvidenceValidationError as exc:
                    raise EvidenceValidationError(
                        f"{candidate.source.title}, page {candidate.page_number}, "
                        f"{candidate.source_table_reference or 'unlabelled table'}, "
                        f"{candidate.measurement.source_value_text!r}: {exc}"
                    ) from exc
                grouped[document_id].append(candidate)
        return grouped, review_items

    def _table_candidates(
        self,
        *,
        table,
        source: DocumentSource,
        cells: list[dict],
        header_rows: set[int],
        raw_evidence: dict[tuple[int, int], sqlite3.Row],
        source_table_reference: str | None,
    ) -> tuple[list[BenchmarkCandidate], list[BenchmarkReviewItem]]:
        table_id = int(table["id"])
        document_id = int(table["document_id"])
        page_number = int(table["page_number"])
        header_cells = [cell for cell in cells if int(cell["row_index"]) in header_rows]
        header_text = " ".join(
            cell["text"].strip() for cell in header_cells if cell.get("text", "").strip()
        )
        table_context = " ".join(
            text
            for text in (
                source.title,
                _table_heading_context(table["heading_context"]),
                header_text,
                " ".join(cell.get("text", "") for cell in cells),
            )
            if text
        )
        table_kind = _record_kind(table_context, header_text)
        predicted_table = bool(
            re.search(r"\bpredict(?:ed|ion)\b", table_context, re.I)
            and re.search(r"\b(?:receiver|receptor)\b", header_text, re.I)
        )
        horizontal_bands = _horizontal_bands(table, cells, header_rows)
        frequency_columns = _vertical_frequency_columns(
            header_cells, cells, header_rows
        )
        all_rows = sorted({int(cell["row_index"]) for cell in cells})
        candidates: list[BenchmarkCandidate] = []
        review_items: list[BenchmarkReviewItem] = []
        seen: set[tuple[int, int]] = set()

        def cite(raw_cell: dict, role: str) -> SourceEvidenceReference | None:
            evidence = raw_evidence.get(
                (int(raw_cell["source_row_index"]), int(raw_cell["source_column_index"]))
            )
            if evidence is None or evidence["evidence_status"] != "present":
                return None
            return SourceEvidenceReference(
                evidence_id=int(evidence["id"]),
                role=role,
                document_id=document_id,
                page_number=int(evidence["page_number"]),
                table_id=table_id,
                row_index=int(evidence["row_index"]),
                column_index=int(evidence["column_index"]),
                quote=evidence["evidence_text"],
            )

        for row_index in all_rows:
            if row_index in header_rows:
                continue
            row_cells = [
                cell
                for cell in cells
                if int(cell["row_index"]) == row_index
                and cell.get("column_start") is not None
                and cell.get("column_end") is not None
                and cell.get("text", "").strip()
            ]
            for cell in row_cells:
                column_start = int(cell["column_start"])
                column_end = int(cell["column_end"])
                if column_start != column_end or column_start in frequency_columns:
                    continue
                value_source = cite(cell, "measurement_value")
                if value_source is None:
                    continue

                column_headers = _column_headers(
                    header_cells, column_start, column_end
                )
                column_context = " ".join(column_headers)
                row_context = " ".join(
                    item.get("text", "").strip() for item in row_cells
                )
                band = _frequency_band(
                    cell=cell,
                    row_cells=row_cells,
                    table_cells=cells,
                    column_headers=column_headers,
                    header_cells=header_cells,
                    horizontal_bands=horizontal_bands,
                    vertical_frequency_columns=frequency_columns,
                    raw_evidence=raw_evidence,
                    table_id=table_id,
                    document_id=document_id,
                    page_number=page_number,
                )
                kind = _record_kind(
                    " ".join((table_context, column_context)),
                    column_context,
                    row_context=row_context,
                    table_fallback=(
                        table_kind
                        if table_kind != "regulatory_noise_criterion"
                        or band is not None
                        or re.search(
                            r"\b(?:criterion|criteria|limits?|threshold|objective|permitted)\b",
                            column_context,
                            re.I,
                        )
                        else None
                    ),
                )
                if (
                    predicted_table
                    and re.search(r"\bpredict(?:ed|ion)\b", column_context, re.I)
                    and not re.search(
                        r"\b(?:criterion|criteria|limits?|margin|compliance|objective)\b",
                        column_context,
                        re.I,
                    )
                ):
                    kind = "predicted_receptor_noise_level"
                if not _is_measurement_cell(
                    cell,
                    column_context,
                    table_context,
                    kind,
                    predicted_table,
                    band,
                ):
                    continue

                parsed = parse_source_value(cell["text"])
                if parsed is None:
                    if re.search(r"\d", cell["text"]):
                        review_items.append(
                            _review_item(
                                source,
                                table,
                                source_table_reference,
                                "ambiguous_numeric_cell",
                                cell["text"],
                                row_index=int(cell["source_row_index"]),
                                column_index=int(cell["source_column_index"]),
                            )
                        )
                    continue
                location_key = (
                    int(cell["source_row_index"]),
                    int(cell["source_column_index"]),
                )
                if location_key in seen:
                    continue
                seen.add(location_key)

                entity_role, entity_label, entity_cell = _row_entity(
                    row_cells,
                    cell,
                    header_cells,
                    column_start,
                )
                citations = [value_source]
                page_context_items = list(_page_context_evidence(
                    table["page_text"],
                    table["heading_context"],
                    document_id,
                    page_number,
                ))
                for context_cell in row_cells:
                    if context_cell is cell or not (
                        _metric_type(context_cell.get("text", "")) != "unknown"
                        or _unit_text(context_cell.get("text", ""))
                    ):
                        continue
                    quote = _exact_whitespace_match(
                        table["page_text"], context_cell["text"]
                    )
                    if quote and not any(
                        existing.quote == quote for existing in page_context_items
                    ):
                        page_context_items.append(
                            PageContextEvidence(
                                document_id=document_id,
                                page_number=page_number,
                                role="nearby_context",
                                quote=quote,
                            )
                        )
                page_context = tuple(page_context_items)
                metric_context = " ".join(
                    [column_context, row_context]
                    + [item.quote for item in page_context]
                )
                metric_type = _metric_type(metric_context)
                bandwidth_text = _bandwidth_text(
                    " ".join((table_context, column_context))
                )
                if band is not None:
                    citations.extend(band[1])

                def add_citation(raw_cell: dict | None, role: str) -> None:
                    if raw_cell is None:
                        return
                    citation = cite(raw_cell, role)
                    if citation is not None and not any(
                        existing.evidence_id == citation.evidence_id
                        and existing.role == citation.role
                        for existing in citations
                    ):
                        citations.append(citation)

                value_unit = _unit_text(cell["text"])
                unit_text = (
                    value_unit
                    or _unit_text(column_context)
                    or _unit_text(row_context)
                    or _unit_text(table["heading_context"])
                )
                unit_source: str | None = None
                if value_unit:
                    unit_source = cell["text"]
                    add_citation(cell, "unit")
                elif _unit_text(row_context):
                    unit_cell = next(
                        (
                            item
                            for item in row_cells
                            if _unit_text(item.get("text", ""))
                        ),
                        None,
                    )
                    if unit_cell is not None:
                        unit_text = _unit_text(unit_cell["text"])
                        unit_source = (
                            unit_text
                            if unit_text and unit_text in unit_cell["text"]
                            else unit_cell["text"]
                        )
                        add_citation(unit_cell, "unit")
                else:
                    unit_cell = next(
                        (
                            header
                            for header in _matching_header_cells(
                                header_cells, column_start, column_end
                            )
                            if _unit_text(header["text"])
                        ),
                        None,
                    )
                    if unit_cell is not None:
                        unit_text = _unit_text(unit_cell["text"])
                        unit_source = (
                            unit_text
                            if unit_text and unit_text in unit_cell["text"]
                            else unit_cell["text"]
                        )
                        add_citation(unit_cell, "unit")
                    elif unit_text:
                        context_item = _page_context(
                            table["page_text"],
                            table["heading_context"],
                            document_id,
                            page_number,
                        )
                        if context_item and unit_text in context_item.quote:
                            unit_source = context_item.quote

                weighting = _weighting(unit_text or "")
                for header in _matching_header_cells(
                    header_cells, column_start, column_end
                ):
                    header_value = header["text"].strip()
                    if _metric_descriptor(header_value):
                        add_citation(header, "metric_context")
                    role = _context_role(header_value)
                    if role:
                        add_citation(header, role)
                    if re.search(
                        r"\bper[\s-]+unit\b|\beach\s+unit\b|\baggregate\b|"
                        r"\bfacility[- ]wide\b|\btotal\s+(?:facility|site)\b",
                        header_value,
                        re.I,
                    ):
                        add_citation(header, "scope")
                    if re.search(
                        r"\b(?:measured|measurement|monitoring|predicted|"
                        r"manufacturer[- ]supplied|modelled|modeled)\b",
                        header_value,
                        re.I,
                    ):
                        add_citation(header, "basis")
                if kind == "regulatory_noise_criterion":
                    for criteria_header in header_cells:
                        if re.search(
                            r"\b(?:criterion|criteria|limits?|threshold|objective|permitted)\b",
                            criteria_header.get("text", ""),
                            re.I,
                        ):
                            add_citation(criteria_header, "metric_context")
                for row_context_cell in row_cells:
                    if re.search(
                        r"\b(?:measured|measurements?|monitoring\s+results?|"
                        r"predicted|modelled|modeled|manufacturer[- ]supplied|"
                        r"provided\s+by\s+(?:the\s+)?manufacturer|"
                        r"supplier\s+noise\s+assessment)\b",
                        row_context_cell.get("text", ""),
                        re.I,
                    ):
                        add_citation(row_context_cell, "basis")
                if entity_cell is not None:
                    role = _entity_evidence_role(entity_role)
                    add_citation(entity_cell, role)
                    entity_column = int(entity_cell["column_start"])
                    for entity_header in _matching_header_cells(
                        header_cells, entity_column, entity_column
                    ):
                        add_citation(entity_header, role)
                if entity_role == "equipment" and entity_label:
                    equipment = EquipmentInformation(source_label=entity_label)
                    quantity = _quantity_cell(row_cells, header_cells, column_start)
                    if quantity is not None:
                        quantity_number = parse_source_value(quantity["text"])
                        if quantity_number and quantity_number.comparator == "exact":
                            equipment = EquipmentInformation(
                                source_label=entity_label,
                                unit_count=int(quantity_number.value)
                                if quantity_number.value
                                and quantity_number.value == quantity_number.value.to_integral_value()
                                else None,
                            )
                            add_citation(quantity, "scope")
                else:
                    equipment = EquipmentInformation()

                record_kind = kind
                relevance_context = " ".join(
                    [source.title, source.original_url]
                    + [item.quote for item in page_context]
                    + [item.quote for item in citations]
                )
                relevance = _bess_relevance(
                    record_kind,
                    relevance_context,
                    entity_label or "",
                    column_context,
                    source.title,
                    source.original_url,
                )
                basis = _value_basis(
                    record_kind,
                    table_context,
                    column_context,
                    row_context,
                )
                aggregation = (
                    "per_unit"
                    if re.search(r"\bper[\s-]+unit\b|\beach\s+unit\b", table_context, re.I)
                    else "unknown"
                )
                descriptor = _metric_descriptor(metric_context)
                frequency = None
                if band is not None:
                    band_info, band_citations = band
                    frequency = band_info
                    for citation in band_citations:
                        if not any(
                            current.evidence_id == citation.evidence_id
                            and current.role == citation.role
                            for current in citations
                        ):
                            citations.append(citation)

                unit_evidence_text = unit_source or unit_text
                if unit_text and not any(
                    unit_evidence_text in item.quote
                    for item in citations + list(page_context)
                ):
                    unit_text = None
                    unit_source = None
                    weighting = "unknown"

                measurement = AcousticMeasurement(
                    value=parsed.value,
                    source_value_text=value_source.quote,
                    comparator=parsed.comparator,
                    lower_bound=parsed.lower_bound,
                    upper_bound=parsed.upper_bound,
                    units=unit_text,
                    source_unit_text=unit_source,
                    metric_descriptor=descriptor,
                    metric_type=metric_type,
                    weighting=weighting,
                    frequency_band=frequency,
                    value_basis=basis,
                    aggregation_basis=aggregation,
                    comparison_eligibility=(
                        "ineligible"
                        if record_kind != "equipment_sound_emission"
                        or relevance != "bess_specific"
                        else "not_assessed"
                    ),
                )
                review_flags = _candidate_review_flags(
                    measurement,
                    record_kind,
                    entity_role,
                    relevance,
                )
                candidate_id = _candidate_id(
                    source,
                    page_number,
                    int(table["table_index"]),
                    cell,
                    record_kind,
                    measurement,
                    entity_role,
                )
                candidate = BenchmarkCandidate(
                    candidate_id=candidate_id,
                    generator_version=GENERATOR_VERSION,
                    source=source,
                    page_number=page_number,
                    table_id=table_id,
                    source_table_reference=source_table_reference,
                    record_kind=record_kind,
                    bess_relevance=relevance,
                    source_entity_role=entity_role,
                    source_entity_label=entity_label,
                    equipment=equipment,
                    measurement=measurement,
                    evidence=tuple(citations),
                    page_context=page_context,
                    review_flags=tuple(review_flags),
                )
                candidates.append(candidate)
                if review_flags:
                    review_items.append(
                        _review_item(
                            source,
                            table,
                            source_table_reference,
                            "candidate_requires_review:" + ",".join(review_flags),
                            cell["text"],
                            row_index=int(cell["source_row_index"]),
                            column_index=int(cell["source_column_index"]),
                        )
                    )
        return candidates, review_items


def parse_source_value(text: str) -> NumericInterpretation | None:
    normalized = text.strip().replace("−", "-")
    if not normalized:
        return None
    bound = _BOUND.fullmatch(normalized)
    if bound:
        operator, number, suffix = bound.groups()
        if suffix and not _is_unit_suffix(suffix):
            return None
        value = _decimal(number)
        if value is None:
            return None
        operator = operator.replace("≤", "<=").replace("≥", ">=")
        if operator in {"<", "<="}:
            return NumericInterpretation(
                "lt" if operator == "<" else "lte",
                upper_bound=value,
            )
        return NumericInterpretation(
            "gt" if operator == ">" else "gte",
            lower_bound=value,
        )
    range_match = _RANGE.fullmatch(normalized)
    if range_match:
        lower_text, upper_text, suffix = range_match.groups()
        if suffix and not _is_unit_suffix(suffix):
            return None
        lower = _decimal(lower_text)
        upper = _decimal(upper_text)
        if lower is None or upper is None or lower > upper:
            return None
        return NumericInterpretation("range", lower_bound=lower, upper_bound=upper)
    exact = _EXACT.fullmatch(normalized)
    if not exact:
        return None
    number, suffix = exact.groups()
    if suffix and not _is_unit_suffix(suffix):
        return None
    value = _decimal(number)
    return NumericInterpretation("exact", value=value) if value is not None else None


def _decimal(value: str) -> Decimal | None:
    try:
        result = Decimal(value)
    except InvalidOperation:
        return None
    return result if result.is_finite() else None


def _is_unit_suffix(value: str) -> bool:
    return bool(
        re.fullmatch(
            r"\s*(?:dB(?:\s*\([ABCZ]\)|[ABCZ]|\s+re\s+.*)?|"
            r"Hz|kHz|Pa|m|ft|feet|%|)\s*",
            value,
            re.I,
        )
    )


def _unit_text(text: str) -> str | None:
    match = _UNIT.search(text)
    return re.sub(r"\s+", " ", match.group(0)).strip() if match else None


def _weighting(unit: str) -> str:
    if re.search(r"dB\s*\(\s*A\s*\)|dBA\b", unit, re.I):
        return "A"
    if re.search(r"dB\s*\(\s*C\s*\)|dBC\b", unit, re.I):
        return "C"
    if re.search(r"dB\s*\(\s*Z\s*\)|dBZ\b", unit, re.I):
        return "Z"
    return "unknown"


def _record_kind(
    context: str,
    column_context: str,
    *,
    row_context: str = "",
    table_fallback: str | None = None,
) -> str:
    combined = " ".join((context, column_context, row_context))
    local_context = " ".join((column_context, row_context))
    caption_context = _table_caption_context(context) or context
    if re.search(
        r"\b(?:Rw\s*\+\s*Ctr|façade|facade|sound\s+insulation|"
        r"building\s+sound|SRI)\b",
        combined,
        re.I,
    ):
        return "building_sound_insulation_performance"
    if re.search(
        r"\b(?:insertion\s+loss|attenuation|mitigation|silencer|"
        r"noise\s+barrier|sound\s+reduction\s+performance)\b",
        combined,
        re.I,
    ):
        return "noise_mitigation_performance"
    if re.search(
        r"\bpredict(?:ed|ion|ive)\b", local_context + " " + caption_context, re.I
    ) and re.search(
        r"\b(?:receiver|receptor|property\s+boundary|property\s+line|"
        r"nearest\s+residence|NSR)\b",
        local_context,
        re.I,
    ):
        return "predicted_receptor_noise_level"
    if _GENERIC_REFERENCE.search(combined):
        return "general_acoustic_reference_example"
    if re.search(
        r"\b(?:criterion|criteria|limits?|threshold|objective|permitted|requirements?|"
        r"shall\s+not\s+exceed|maximum\s+allowable)\b",
        local_context + " " + caption_context,
        re.I,
    ):
        return "regulatory_noise_criterion"
    if re.search(r"\bsound\s+power\b|\bSWL\b", caption_context + " " + column_context, re.I) and (
        _EQUIPMENT_WORDS.search(row_context)
        or re.search(r"\b(?:equipment|plant\s+item|source|item)\b", column_context, re.I)
    ):
        return "equipment_sound_emission"
    if re.search(
        r"\b(?:monitoring|ambient|background|measured|measurement\s+results?)\b",
        local_context + " " + caption_context,
        re.I,
    ):
        return "ambient_environmental_measurement"
    return table_fallback or "unknown_or_ambiguous"


def _table_caption_context(context: str) -> str:
    references = list(_TABLE_REFERENCE.finditer(context))
    if not references:
        return ""
    tail = context[references[-1].start() :]
    return re.split(r"[\r\n]", tail, maxsplit=1)[0][:300]


def _table_heading_context(heading: str) -> str:
    lines = [line.strip() for line in heading.splitlines() if line.strip()]
    for line in reversed(lines):
        if re.match(r"^table\s+\d+(?:[-–]\d+)?\b", line, re.I):
            return line
    tail = heading[-400:].strip()
    sentence_starts = list(re.finditer(r"(?<=[.!?])\s+", tail))
    if sentence_starts:
        tail = tail[sentence_starts[-1].end() :]
    return tail


def _metric_type(context: str) -> str:
    for metric, pattern in _METRIC_PATTERNS:
        if pattern.search(context):
            return metric
    return "unknown"


def _is_measurement_cell(
    cell: dict,
    column_context: str,
    table_context: str,
    record_kind: str,
    predicted_table: bool,
    band: tuple[FrequencyBandInformation, list[SourceEvidenceReference]] | None,
) -> bool:
    text = cell.get("text", "").strip()
    if not text or parse_source_value(text) is None:
        return False
    if _EXCLUDED_COLUMN.search(column_context):
        if band is None:
            return False
    if predicted_table and re.search(
        r"\b(?:criterion|criteria|margin|compliance|objective)\b",
        column_context,
        re.I,
    ):
        return False
    explicit_column = bool(
        re.search(
            r"\bdB\b|sound|noise|acoustic|SPL|SWL|SRI|insertion\s+loss|"
            r"\bL\s*[_\s]?(?:Aeq|AF90|A90|AFmax|Amax|den|night|day)\b",
            column_context,
            re.I,
        )
    )
    if explicit_column:
        return True
    if band is not None and _ACOUSTIC.search(table_context):
        return True
    return bool(
        record_kind == "regulatory_noise_criterion"
        and _ACOUSTIC.search(table_context)
        and (band is not None or _unit_text(table_context))
    )


def _column_headers(
    headers: list[dict],
    column_start: int,
    column_end: int,
) -> list[str]:
    return [
        cell["text"].strip()
        for cell in _matching_header_cells(headers, column_start, column_end)
        if cell.get("text", "").strip()
    ]


def _matching_header_cells(
    headers: list[dict],
    column_start: int,
    column_end: int,
) -> list[dict]:
    return [
        header
        for header in headers
        if header.get("column_start") is not None
        and header.get("column_end") is not None
        and int(header["column_start"]) <= column_end
        and int(header["column_end"]) >= column_start
        and header.get("text", "").strip()
    ]


def _horizontal_bands(table, cells: list[dict], header_rows: set[int]) -> dict[int, dict]:
    result = {}
    for band in json.loads(table["frequency_bands_json"]):
        if (
            band.get("logical_column") is not None
            and int(band.get("source_row_index", -1)) in header_rows
        ):
            result[int(band["logical_column"])] = band
    if result:
        return result
    header_text = " ".join(
        cell.get("text", "")
        for cell in cells
        if int(cell["row_index"]) in header_rows
    )
    if not _FREQUENCY_HEADER.search(header_text):
        return result
    rows: dict[int, list[dict]] = defaultdict(list)
    for cell in cells:
        rows[int(cell["row_index"])].append(cell)
    first_data_row = min(
        (
            row_index
            for row_index, row_cells in rows.items()
            if row_index > max(header_rows, default=-1)
            and any(
                int(cell.get("column_start", -1)) == 0
                and cell.get("text", "").strip()
                and _frequency_hz(cell["text"], header_text) is None
                for cell in row_cells
            )
        ),
        default=max(rows, default=-1) + 1,
    )
    possible_rows = set(header_rows)
    possible_rows.update(
        row_index
        for row_index in rows
        if max(header_rows, default=-1) < row_index < first_data_row
    )
    for row_index in possible_rows:
        frequency_cells = [
            cell
            for cell in rows.get(row_index, [])
            if cell.get("column_start") is not None
            and cell.get("column_start") == cell.get("column_end")
            and _frequency_hz(cell.get("text", ""), header_text) is not None
        ]
        if len(frequency_cells) < 3:
            continue
        for cell in frequency_cells:
            center = _frequency_hz(cell["text"].strip(), header_text)
            if center is not None:
                result[int(cell["column_start"])] = {
                    "center_hz": str(center),
                    "source_label": cell["text"].strip(),
                    "source_row_index": int(cell["source_row_index"]),
                    "logical_column": int(cell["column_start"]),
                }
    return result


def _vertical_frequency_columns(
    header_cells: list[dict],
    cells: list[dict],
    header_rows: set[int],
) -> set[int]:
    result = set()
    for header in header_cells:
        start = header.get("column_start")
        end = header.get("column_end")
        if (
            start is None
            or end is None
            or int(start) != int(end)
            or not _FREQUENCY_HEADER.search(header.get("text", ""))
        ):
            continue
        header_context = " ".join(
            cell.get("text", "")
            for cell in header_cells
            if int(cell.get("column_start", -1)) == int(start)
        )
        if not re.search(r"\b(?:Hz|kHz|hertz|kilohertz)\b", header_context, re.I):
            continue
        if any(
            int(cell["row_index"]) not in header_rows
            and cell.get("column_start") is not None
            and int(cell["column_start"]) == int(start)
            and _frequency_hz(cell.get("text", ""), header_context) is not None
            for cell in cells
        ):
            result.add(int(start))
    return result


def _frequency_band(
    *,
    cell: dict,
    row_cells: list[dict],
    table_cells: list[dict],
    column_headers: list[str],
    header_cells: list[dict],
    horizontal_bands: dict[int, dict],
    vertical_frequency_columns: set[int],
    raw_evidence: dict[tuple[int, int], sqlite3.Row],
    table_id: int,
    document_id: int,
    page_number: int,
) -> tuple[FrequencyBandInformation, list[SourceEvidenceReference]] | None:
    logical_column = int(cell["column_start"])
    mapping = horizontal_bands.get(logical_column)
    if mapping is not None:
        label = str(mapping["source_label"])
        source_cell = next(
            (
                item
                for item in table_cells
                if int(item["row_index"]) == int(mapping["source_row_index"])
                and item.get("column_start") is not None
                and int(item["column_start"]) <= logical_column
                <= int(item.get("column_end", item["column_start"]))
            ),
            None,
        )
        citations = []
        if source_cell is not None:
            citation = _citation_from_cell(
                source_cell,
                raw_evidence,
                document_id,
                page_number,
                table_id,
                "frequency_band",
            )
            if citation:
                citations.append(citation)
        for header in header_cells:
            if (
                re.search(r"\bfrequency\b|\bfreq\.?\b|octave[- ]band", header.get("text", ""), re.I)
                or re.search(
                    r"\b(?:Hz|kHz|hertz|kilohertz)\b",
                    header.get("text", ""),
                    re.I,
                )
            ):
                header_citation = _citation_from_cell(
                    header,
                    raw_evidence,
                    document_id,
                    page_number,
                    table_id,
                    "frequency_band",
                )
                if header_citation and not any(
                    item.evidence_id == header_citation.evidence_id
                    for item in citations
                ):
                    citations.append(header_citation)
        source_bandwidth_text = next(
            (
                header
                for header in column_headers
                if re.search(r"one[- ]third octave|octave", header, re.I)
            ),
            None,
        )
        bandwidth = _bandwidth(source_bandwidth_text)
        return (
            FrequencyBandInformation(
                center_hz=Decimal(str(mapping["center_hz"])),
                source_label=label,
                bandwidth=bandwidth,
                source_bandwidth_text=source_bandwidth_text,
            ),
            citations,
        )

    if not vertical_frequency_columns:
        return None
    for possible in row_cells:
        if int(possible["column_start"]) not in vertical_frequency_columns:
            continue
        possible_headers = _column_headers(
            header_cells,
            int(possible["column_start"]),
            int(possible.get("column_end", possible["column_start"])),
        )
        header_text = " ".join(possible_headers)
        if not _FREQUENCY_HEADER.search(header_text):
            continue
        center = _frequency_hz(possible["text"], header_text)
        if center is None:
            continue
        citations = []
        citation = _citation_from_cell(
            possible,
            raw_evidence,
            document_id,
            page_number,
            table_id,
            "frequency_band",
        )
        if citation:
            citations.append(citation)
        frequency_header = next(
            (h for h in header_cells if h.get("text", "") in possible_headers),
            None,
        )
        if frequency_header is not None:
            header_citation = _citation_from_cell(
                frequency_header,
                raw_evidence,
                document_id,
                page_number,
                table_id,
                "frequency_band",
            )
            if header_citation:
                citations.append(header_citation)
        bandwidth_text = next(
            (
                text
                for text in possible_headers
                if re.search(r"one[- ]third octave|octave", text, re.I)
            ),
            header_text or None,
        )
        return (
            FrequencyBandInformation(
                center_hz=center,
                source_label=possible["text"].strip(),
                bandwidth=_bandwidth(bandwidth_text),
                source_bandwidth_text=bandwidth_text,
            ),
            citations,
        )
    return None


def _citation_from_cell(
    cell: dict,
    raw_evidence: dict[tuple[int, int], sqlite3.Row],
    document_id: int,
    page_number: int,
    table_id: int,
    role: str,
) -> SourceEvidenceReference | None:
    evidence = raw_evidence.get(
        (int(cell["source_row_index"]), int(cell["source_column_index"]))
    )
    if evidence is None or evidence["evidence_status"] != "present":
        return None
    return SourceEvidenceReference(
        evidence_id=int(evidence["id"]),
        role=role,
        document_id=document_id,
        page_number=page_number,
        table_id=table_id,
        row_index=int(evidence["row_index"]),
        column_index=int(evidence["column_index"]),
        quote=evidence["evidence_text"],
    )


def _frequency_hz(text: str, context: str) -> Decimal | None:
    match = _FREQUENCY.fullmatch(text.strip())
    if match is None:
        return None
    amount = _decimal(match.group(1))
    if amount is None or amount <= 0:
        return None
    unit = match.group(2)
    if not unit:
        unit_match = re.search(r"\b(kHz|Hz|kilohertz|hertz)\b", context, re.I)
        if not unit_match:
            return None
        unit = unit_match.group(1)
    return (
        amount * Decimal("1000")
        if unit.casefold() in {"khz", "kilohertz", "k"}
        else amount
    )


def _bandwidth(text: str | None) -> str:
    if not text:
        return "unknown"
    if re.search(r"one[- ]third octave|1\s*/\s*3\s*octave", text, re.I):
        return "one_third_octave"
    if re.search(r"octave", text, re.I):
        return "octave"
    if re.search(r"narrowband", text, re.I):
        return "narrowband"
    return "unknown"


def _bandwidth_text(context: str) -> str | None:
    match = re.search(
        r"(?:one[- ]third|1\s*/\s*3)\s*octave(?:[- ]band)?|"
        r"octave[- ]band|narrowband",
        context,
        re.I,
    )
    return match.group(0) if match else None


def _unit_text(text: str) -> str | None:
    match = _UNIT.search(text)
    return re.sub(r"\s+", " ", match.group(0)).strip() if match else None


def _metric_descriptor(context: str) -> str | None:
    patterns = (
        r"L\s*,?\s*dB\s*\([ABC]\)\s*(?:Aeq|AF90|A90|AFmax|Amax)",
        r"\b(?:LAeq|LAF90|LA90|LAFmax|LAmax|Leq(?:\s+\d+\s*min)?)\b",
        r"\bRw\s*\+\s*Ctr\b",
    )
    for pattern in patterns:
        match = re.search(pattern, context, re.I)
        if match:
            return match.group(0)
    return None


def _context_role(header: str) -> str | None:
    if re.search(r"sound\s+power|sound\s+pressure|\bSPL\b", header, re.I):
        return "metric_context"
    if re.search(r"\bmeasured\b|\bmonitoring\b", header, re.I):
        return "basis"
    if re.search(r"\bpredict(?:ed|ion)\b|\bmodelled\b|\bmodeled\b", header, re.I):
        return "basis"
    if re.search(r"manufacturer[- ]supplied|supplier|manufacturer", header, re.I):
        return "basis"
    if re.search(r"\bdB\b|Hz|kHz", header, re.I):
        return "unit"
    return None


def _row_entity(
    row_cells: list[dict],
    value_cell: dict,
    headers: list[dict],
    value_column: int,
) -> tuple[str, str | None, dict | None]:
    candidates: list[tuple[int, str, str, dict]] = []
    for cell in row_cells:
        if cell is value_cell or not cell.get("text", "").strip():
            continue
        column_start = int(cell["column_start"])
        column_end = int(cell.get("column_end", column_start))
        if column_start <= value_column <= column_end:
            continue
        header = " ".join(_column_headers(headers, column_start, column_end))
        text = cell["text"].strip()
        if text.casefold() in {"-", "--", "—", "–", "n/a", "na", "none"}:
            continue
        role = None
        label = text
        priority = 99
        if re.search(r"\b(?:equipment|plant\s+item|item)\b", header, re.I) and (
            _EQUIPMENT_WORDS.search(text)
            or re.search(r"\b(?:equipment|plant\s+item|item)\b", header, re.I)
        ):
            role, priority = "equipment", 0
        elif re.search(r"monitoring\s+location", header, re.I):
            role, priority = "monitoring_location", 1
            label = f"Monitoring Location {text}"
        elif re.search(r"\b(?:receiver|receptor)\b", header, re.I):
            role, priority = "receiver", 2
        elif re.search(r"\b(?:time|period|date)\b", header, re.I):
            role, priority = "time_period", 6
        elif re.search(r"\benvironment\b|\bdescription\b", header, re.I):
            role, priority = "environment_description", 4
        elif re.search(r"\b(?:zone|category|regulatory|land use)\b", header, re.I):
            role, priority = "regulatory_category", 5
        elif re.search(r"\b(?:configuration|façade|facade|building|element)\b", header, re.I):
            role, priority = "building_configuration", 3
        elif re.search(r"\b(?:sound\s+)?source\b|\bequipment\b", header, re.I) and re.fullmatch(
            r"(?:BESS|battery\s+energy\s+storage(?:\s+system)?)", text, re.I
        ):
            role, priority = "equipment", 0
        elif re.search(r"\bnoise\s+source\b", header, re.I) and re.search(
            r"\b(?:limit|limits|criteria|criterion|code)\b", text, re.I
        ):
            role, priority = "regulatory_category", 5
        elif re.search(r"\bnoise\s+source\b", header, re.I) and re.search(
            r"\b(?:predicted|property\s+boundary|nearest\s+residence)\b",
            text,
            re.I,
        ):
            role, priority = "receiver", 2
        elif _EQUIPMENT_WORDS.search(text):
            role, priority = "equipment", 0
        elif re.fullmatch(r"\d{1,2}:\d{2}(?:\s*[ap]\.?m\.?)?", text, re.I):
            role, priority = "time_period", 6
        elif _GENERIC_REFERENCE.search(text):
            role, priority = "environment_description", 4
        if role:
            candidates.append((priority, role, label, cell))
    if not candidates:
        return "unknown", None, None
    _, role, label, cell = min(candidates, key=lambda item: item[0])
    return role, label, cell


def _entity_evidence_role(role: str) -> str:
    return {
        "equipment": "equipment_label",
        "noise_source": "equipment_label",
        "receiver": "receiver_label",
        "monitoring_location": "monitoring_location",
        "time_period": "time_period",
        "environment_description": "environment_description",
        "regulatory_category": "regulatory_category",
        "building_configuration": "building_configuration",
    }.get(role, "metric_context")


def _quantity_cell(
    row_cells: list[dict],
    headers: list[dict],
    equipment_column: int,
) -> dict | None:
    for cell in row_cells:
        column = int(cell["column_start"])
        if column == equipment_column:
            continue
        header = " ".join(_column_headers(headers, column, int(cell.get("column_end", column))))
        if re.search(r"\b(?:quantity|qty|number\s+of\s+units?)\b", header, re.I):
            return cell
    return None


def _value_basis(
    record_kind: str,
    table_context: str,
    column_context: str,
    row_context: str,
) -> str:
    local_context = " ".join(
        (column_context, row_context, _table_caption_context(table_context))
    )
    if record_kind == "predicted_receptor_noise_level" or re.search(
        r"\bpredict(?:ed|ion)\b", column_context, re.I
    ):
        return "predicted"
    if re.search(
        r"\bmanufacturer[- ]supplied\b|\bprovided\s+by\s+(?:the\s+)?manufacturer\b|"
        r"\bsupplier\s+noise\s+assessment\b",
        local_context,
        re.I,
    ):
        return "manufacturer_supplied"
    if record_kind == "ambient_environmental_measurement" and re.search(
        r"\bmeasured\b|\bmonitoring\s+results?\b|\bmeasurement\s+results?\b",
        local_context,
        re.I,
    ):
        return "measured"
    if re.search(r"\bmeasured\b|\bnoise\s+measurements?\b", local_context, re.I):
        return "measured"
    return "unknown"


def _bess_relevance(
    record_kind: str,
    table_context: str,
    entity_label: str,
    column_context: str,
    document_title: str,
    source_url: str,
) -> str:
    document_context = " ".join((document_title, source_url))
    finding_context = " ".join((table_context, entity_label, column_context))
    if (
        record_kind == "general_acoustic_reference_example"
        or _GENERIC_REFERENCE.search(finding_context)
    ):
        return "general_reference"
    direct_bess_evidence = bool(
        _BESS.search(document_context)
        or _BESS.search(entity_label)
        or _BESS_LINKED_CONTEXT.search(finding_context)
    )
    if direct_bess_evidence:
        if record_kind == "regulatory_noise_criterion":
            return "bess_regulatory_context"
        if record_kind in {
            "equipment_sound_emission",
            "ambient_environmental_measurement",
            "predicted_receptor_noise_level",
        }:
            return "bess_specific"
        return "uncertain"
    if record_kind in {
        "building_sound_insulation_performance",
        "noise_mitigation_performance",
    }:
        return "cross_industry"
    return "uncertain"


def _candidate_review_flags(
    measurement: AcousticMeasurement,
    record_kind: str,
    entity_role: str,
    relevance: str,
) -> list[str]:
    flags = []
    if record_kind == "unknown_or_ambiguous":
        flags.append("record_kind_unknown")
    if entity_role == "unknown":
        flags.append("source_entity_role_unknown")
    if measurement.metric_type == "unknown":
        flags.append("metric_type_unknown")
    if not measurement.units:
        flags.append("unit_missing")
    if measurement.value_basis == "unknown":
        flags.append("measurement_basis_unknown")
    if measurement.comparator != "exact":
        flags.append("non_exact_numeric_interpretation")
    if relevance in {"uncertain", "unrelated"}:
        flags.append("bess_relevance_uncertain")
    return flags


def _candidate_id(
    source: DocumentSource,
    page_number: int,
    table_index: int,
    cell: dict,
    record_kind: str,
    measurement: AcousticMeasurement,
    entity_role: str,
) -> str:
    identity = {
        "version": GENERATOR_VERSION,
        "document_id": source.document_id,
        "sha256": source.sha256,
        "page_number": page_number,
        "table_index": table_index,
        "source_row": int(cell["source_row_index"]),
        "source_column": int(cell["source_column_index"]),
        "record_kind": record_kind,
        "entity_role": entity_role,
        "measurement": measurement.model_dump(mode="json"),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return f"cand-{digest[:16]}"


def _table_reference(heading: str, page_text: str, cells: list[dict]) -> str | None:
    for line in reversed(heading.splitlines()):
        match = re.match(r"\s*(table\s+\d+(?:[-–]\d+)?)\b", line, re.I)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
    return None


def _page_context(
    page_text: str,
    heading_context: str,
    document_id: int,
    page_number: int,
) -> PageContextEvidence | None:
    table_reference = _table_reference(heading_context, "", [])
    if table_reference:
        page_caption_lines = [
            line.strip()
            for line in page_text.splitlines()
            if re.match(
                r"^\s*" + re.escape(table_reference) + r"\b",
                line,
                re.I,
            )
        ]
        if page_caption_lines:
            caption = next(
                (
                    line
                    for line in reversed(page_caption_lines)
                    if _ACOUSTIC.search(line)
                ),
                page_caption_lines[-1],
            )
            return PageContextEvidence(
                document_id=document_id,
                page_number=page_number,
                role="caption",
                quote=caption,
            )
    else:
        regulatory_context_lines = [
            line.strip()
            for line in page_text.splitlines()
            if _ACOUSTIC.search(line)
            and re.search(
                r"\b(?:criterion|criteria|limits?|threshold|objective|permitted)\b",
                line,
                re.I,
            )
        ]
        if regulatory_context_lines:
            return PageContextEvidence(
                document_id=document_id,
                page_number=page_number,
                role="nearby_context",
                quote=regulatory_context_lines[-1],
            )
    candidates = [
        line.strip()
        for line in heading_context.splitlines()
        if line.strip()
    ]
    caption_candidates = [
        line
        for line in candidates
        if re.match(r"^table\s+\d+(?:[-–]\d+)?\b", line, re.I) and len(line) > 10
    ]
    other_candidates = [line for line in candidates if line not in caption_candidates]
    candidates = list(reversed(caption_candidates)) + sorted(
        other_candidates,
        key=lambda line: -len(line),
    )
    for text in candidates:
        quote = _exact_whitespace_match(page_text, text)
        if quote:
            role = "caption" if _TABLE_REFERENCE.search(quote) else "nearby_context"
            return PageContextEvidence(
                document_id=document_id,
                page_number=page_number,
                role=role,
                quote=quote,
            )
    table_match = _TABLE_REFERENCE.search(heading_context)
    if table_match:
        page_match = re.search(re.escape(table_match.group(0)), page_text, re.I)
        if page_match:
            line_start = page_text.rfind("\n", 0, page_match.start()) + 1
            line_end = page_text.find("\n", page_match.end())
            if line_end < 0:
                line_end = len(page_text)
            next_line_end = page_text.find("\n", line_end + 1)
            if next_line_end >= 0 and next_line_end - line_start <= 400:
                line_end = next_line_end
            quote = page_text[line_start:line_end].strip()
            if quote:
                return PageContextEvidence(
                    document_id=document_id,
                    page_number=page_number,
                    role="caption",
                    quote=quote,
                )
    return None


def _page_context_evidence(
    page_text: str,
    heading_context: str,
    document_id: int,
    page_number: int,
) -> tuple[PageContextEvidence, ...]:
    result = []
    caption = _page_context(page_text, heading_context, document_id, page_number)
    if caption is not None:
        if caption.role == "caption":
            start = page_text.rfind(caption.quote)
            if start >= 0:
                lines = page_text[start:].splitlines()
                selected = []
                for line in lines:
                    if not line.strip() and selected:
                        break
                    if _is_likely_table_data_row(line):
                        break
                    selected.append(line)
                    if sum(len(item) for item in selected) >= 600:
                        break
                quote = "\n".join(selected).strip()
                if quote:
                    caption = PageContextEvidence(
                        document_id=document_id,
                        page_number=page_number,
                        role="caption",
                        quote=quote,
                    )
        result.append(caption)
    table_reference = _table_reference(heading_context, "", [])
    if table_reference:
        table_matches = list(
            re.finditer(re.escape(table_reference), page_text, re.I)
        )
        table_position = table_matches[-1] if table_matches else None
        if table_position:
            bess_match = min(
                (
                    match
                    for match in _BESS.finditer(page_text)
                    if abs(match.start() - table_position.start()) <= 800
                ),
                key=lambda match: abs(match.start() - table_position.start()),
                default=None,
            )
            if bess_match:
                start = page_text.rfind("\n", 0, bess_match.start()) + 1
                end = page_text.find("\n", bess_match.end())
                if end < 0:
                    end = len(page_text)
                quote = page_text[start:end].strip()
                if quote and not any(item.quote == quote for item in result):
                    result.append(
                        PageContextEvidence(
                            document_id=document_id,
                            page_number=page_number,
                            role="nearby_context",
                            quote=quote,
                        )
                    )
    context_start = (
        page_text.rfind(result[0].quote) if result else -1
    )
    for line in page_text.splitlines():
        if not re.search(r"\b(?:decibels?|dB|Hz|kHz)\b", line, re.I):
            continue
        line_position = page_text.find(line, max(context_start, 0))
        if context_start >= 0 and abs(line_position - context_start) > 800:
            continue
        quote = line.strip()
        if quote and not any(item.quote == quote for item in result):
            result.append(
                PageContextEvidence(
                    document_id=document_id,
                    page_number=page_number,
                    role="nearby_context",
                    quote=quote,
                )
            )
    return tuple(result)


def _is_likely_table_data_row(text: str) -> bool:
    numeric_cells = re.findall(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)", text)
    if len(numeric_cells) < 3 or _FREQUENCY_HEADER.search(text):
        return False
    return bool(re.search(r"\b[A-Za-z]{3,}\b", text))


def _exact_whitespace_match(source: str, target: str) -> str | None:
    normalized = re.sub(r"\s+", " ", target.strip())
    if not normalized:
        return None
    pattern = re.escape(normalized).replace(r"\ ", r"\s+")
    match = re.search(pattern, source, re.I)
    return match.group(0) if match else None


def _review_item(
    source: DocumentSource,
    table,
    table_reference: str | None,
    reason: str,
    excerpt: str,
    *,
    row_index: int | None = None,
    column_index: int | None = None,
) -> BenchmarkReviewItem:
    identity = {
        "version": GENERATOR_VERSION,
        "sha256": source.sha256,
        "page": int(table["page_number"]),
        "table_index": int(table["table_index"]),
        "reason": reason,
        "row": row_index,
        "column": column_index,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return BenchmarkReviewItem(
        review_item_id=f"review-{digest[:16]}",
        generator_version=GENERATOR_VERSION,
        source=source,
        page_number=int(table["page_number"]),
        table_id=int(table["id"]),
        table_index=int(table["table_index"]),
        row_index=row_index,
        column_index=column_index,
        source_table_reference=table_reference,
        reason=reason,
        excerpt=excerpt[:800],
    )
