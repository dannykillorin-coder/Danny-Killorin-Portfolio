from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParseWarning:
    code: str
    message: str
    page_number: int | None = None
    review_flag: str | None = None


@dataclass(frozen=True)
class TableCell:
    row_index: int
    column_index: int
    text: str
    bounding_box: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class LogicalTableCell:
    row_index: int
    column_start: int | None
    column_end: int | None
    text: str
    source_row_index: int
    source_column_index: int
    bounding_box: tuple[float, float, float, float] | None = None
    row_start: int | None = None
    row_end: int | None = None


@dataclass(frozen=True)
class FrequencyBand:
    source_label: str
    center_hz: float
    logical_column: int
    source_row_index: int


@dataclass(frozen=True)
class ExtractedTable:
    page_number: int
    table_index: int
    row_count: int
    column_count: int
    cells: tuple[TableCell, ...]
    column_headers: tuple[str, ...]
    header_row_index: int | None
    heading_context: str
    frequency_band_labels: tuple[str, ...]
    units: tuple[str, ...]
    bounding_box: tuple[float, float, float, float] | None
    logical_row_count: int = 0
    logical_column_count: int = 0
    logical_cells: tuple[LogicalTableCell, ...] = ()
    header_row_indices: tuple[int, ...] = ()
    frequency_bands: tuple[FrequencyBand, ...] = ()
    measurement_types: tuple[str, ...] = ()
    measurement_type_evidence: tuple[str, ...] = ()
    alignment_status: str = "ambiguous"
    review_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str
    text_status: str
    text_character_count: int
    image_count: int
    acoustic_signals: tuple[str, ...]
    review_flags: tuple[str, ...]


@dataclass(frozen=True)
class PDFParseResult:
    source_path: Path
    source_sha256: str | None
    parser_version: str
    status: str
    metadata: dict[str, Any]
    pages: tuple[ExtractedPage, ...]
    tables: tuple[ExtractedTable, ...]
    warnings: tuple[ParseWarning, ...]
    review_flags: tuple[str, ...]


@dataclass(frozen=True)
class ClassificationEvidence:
    page_number: int
    matched_term: str
    excerpt: str
    evidence_status: str = "present"


@dataclass(frozen=True)
class DocumentClassification:
    category: str
    score: float
    direct_bess_evidence: bool
    cross_industry: bool
    is_primary: bool
    review_status: str
    evidence: tuple[ClassificationEvidence, ...]


@dataclass(frozen=True)
class ProcessingSummary:
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    review_required: int = 0
    outstanding_review: int = 0
