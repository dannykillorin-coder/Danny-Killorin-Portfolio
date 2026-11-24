from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

import pymupdf

from bai.documents.models import (
    ExtractedPage,
    ExtractedTable,
    FrequencyBand,
    LogicalTableCell,
    PDFParseResult,
    ParseWarning,
    TableCell,
)


PARSER_VERSION = "pymupdf-2"
IMAGE_TEXT_MIN_CHARS = 80
ACOUSTIC_SIGNAL_PATTERN = re.compile(
    r"(?<!\w)(?:dB(?:\s*\([A-Z]\))?|sound\s+(?:pressure|power)|"
    r"(?:1\s*/\s*3|third[- ]octave|octave)[- ]band|"
    r"L(?:Aeq|Amax|AFmax|A90|A10|den|night|day))(?!\w)",
    re.IGNORECASE,
)
MEASUREMENT_CONTEXT_PATTERN = re.compile(
    r"\b(?:noise|acoustic|sound|octave[- ]band|SPL|SRI|"
    r"insertion\s+loss|sound\s+reduction\s+index)\b",
    re.IGNORECASE,
)
TABLE_CANDIDATE_PATTERN = re.compile(
    r"\b(?:table\s+\d+|(?:1\s*/\s*3|third[- ]octave|octave)[- ]band|frequency\s+bands?)\b",
    re.IGNORECASE,
)
FREQUENCY_LABEL_PATTERN = re.compile(
    r"\b(?:frequency|freq\.?|(?:1\s*/\s*3|third[- ]octave|octave)[- ]band|"
    r"band\s+(?:center|centre))\b",
    re.IGNORECASE,
)
UNIT_PATTERN = re.compile(
    r"(?<![A-Za-z])(?:dB\s*\([ABC]\)|dB[ABCL]|dB(?![A-Za-z])|"
    r"kHz|Hz|Hertz|MW|MWh|kW|kWh|m/s|Pa|mm|m)(?![A-Za-z])",
    re.IGNORECASE,
)
FREQUENCY_VALUE_PATTERN = re.compile(
    r"(?<![\w.])(?P<label>\d+(?:\.\d+)?\s*(?:kHz|Hz|k)?)(?![\w])",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")
EXPLICIT_FREQUENCY_SUFFIX_PATTERN = re.compile(r"(?:kHz|Hz|k)$", re.IGNORECASE)
MEASUREMENT_TYPE_PATTERNS = (
    (
        "sound_power_level",
        re.compile(r"\bsound\s+power\s+levels?\b|\bL\s*[_(\[]?\s*W\s*[)\]]?", re.I),
    ),
    (
        "sound_pressure_level",
        re.compile(r"\bsound\s+pressure\s+levels?\b|\bSPL\b", re.I),
    ),
    (
        "sound_reduction_index",
        re.compile(r"\bsound\s+reduction\s+index\b|\bSRI\b", re.I),
    ),
    ("insertion_loss", re.compile(r"\binsertion\s+loss\b", re.I)),
)
SOUND_POWER_REFERENCE_PATTERN = re.compile(
    r"\bdB\s+re\s+(?:10\s*(?:\^\s*[-\N{MINUS SIGN}]?\s*12|[-\N{MINUS SIGN}]\s*12)\s*(?:watts?|W)|"
    r"20\s*(?:\N{MICRO SIGN}|\N{GREEK SMALL LETTER MU}|u)?Pa)\b",
    re.IGNORECASE,
)


class PDFParser:
    def parse(self, path: str | Path) -> PDFParseResult:
        source_path = Path(path).expanduser().resolve()
        try:
            digest = _sha256(source_path)
        except OSError as exc:
            return self._failed(
                source_path,
                None,
                "file_unavailable",
                f"Cannot read PDF file: {exc}",
            )

        pages: list[ExtractedPage] = []
        tables: list[ExtractedTable] = []
        warnings: list[ParseWarning] = []
        document_flags: set[str] = set()
        metadata: dict[str, Any] = {}

        try:
            with pymupdf.open(source_path) as document:
                metadata = _normalize_metadata(document.metadata or {})
                if document.needs_pass:
                    return self._failed(
                        source_path,
                        digest,
                        "password_protected",
                        "PDF requires a password; text extraction was not attempted.",
                        metadata=metadata,
                    )

                for page_index, page in enumerate(document):
                    page_number = page_index + 1
                    try:
                        text = page.get_text("text", sort=True) or ""
                        image_count = len(page.get_images(full=True))
                    except (pymupdf.FileDataError, RuntimeError, ValueError, OSError) as exc:
                        warning = ParseWarning(
                            "page_text_extraction_failed",
                            f"Page text could not be extracted ({type(exc).__name__}).",
                            page_number,
                            "manual_review",
                        )
                        warnings.append(warning)
                        document_flags.add("manual_review")
                        pages.append(
                            ExtractedPage(
                                page_number=page_number,
                                text="",
                                text_status="extraction_failed",
                                text_character_count=0,
                                image_count=0,
                                acoustic_signals=(),
                                review_flags=("manual_review", "content_not_assessed"),
                            )
                        )
                        continue

                    clean_text = text.strip()
                    text_count = len(clean_text)
                    page_flags: set[str] = set()
                    if image_count > 0 and text_count < IMAGE_TEXT_MIN_CHARS:
                        text_status = "image_heavy"
                        page_flags.update(("ocr_review", "content_not_fully_assessed"))
                        document_flags.update(page_flags)
                        warnings.append(
                            ParseWarning(
                                "image_heavy_page",
                                "Page contains images and little extractable text; source content may be unavailable to text parsing.",
                                page_number,
                                "ocr_review",
                            )
                        )
                    elif text_count == 0:
                        text_status = "no_extractable_text"
                        page_flags.update(("manual_review", "content_not_assessed"))
                        document_flags.update(page_flags)
                        warnings.append(
                            ParseWarning(
                                "no_extractable_text",
                                "No selectable text was extracted; this does not establish that the source page is blank.",
                                page_number,
                                "manual_review",
                            )
                        )
                    else:
                        text_status = "extracted"

                    signals = _unique_matches(ACOUSTIC_SIGNAL_PATTERN, clean_text)
                    pages.append(
                        ExtractedPage(
                            page_number=page_number,
                            text=text,
                            text_status=text_status,
                            text_character_count=text_count,
                            image_count=image_count,
                            acoustic_signals=signals,
                            review_flags=tuple(sorted(page_flags)),
                        )
                    )

                    try:
                        page_tables = page.find_tables()
                        for table_index, table in enumerate(page_tables.tables):
                            extracted = _extract_table(
                                page,
                                table,
                                page_number=page_number,
                                table_index=table_index,
                            )
                            if extracted.row_count and extracted.column_count:
                                tables.append(extracted)
                                if extracted.review_flags:
                                    document_flags.update(extracted.review_flags)
                                    warnings.extend(
                                        ParseWarning(
                                            flag,
                                            _table_review_message(flag),
                                            page_number,
                                            "table_review",
                                        )
                                        for flag in extracted.review_flags
                                    )
                        if not page_tables.tables and TABLE_CANDIDATE_PATTERN.search(clean_text):
                            warnings.append(
                                ParseWarning(
                                    "candidate_table_not_detected",
                                    "Text suggests tabular or frequency-band content, but no structured table was detected on this page.",
                                    page_number,
                                    "table_review",
                                )
                            )
                            document_flags.add("table_review")
                    except (pymupdf.FileDataError, RuntimeError, ValueError, OSError) as exc:
                        warnings.append(
                            ParseWarning(
                                "table_extraction_failed",
                                f"Candidate table extraction failed ({type(exc).__name__}).",
                                page_number,
                                "table_review",
                            )
                        )
                        document_flags.add("table_review")

        except (
            pymupdf.FileDataError,
            pymupdf.EmptyFileError,
            RuntimeError,
            ValueError,
            OSError,
        ) as exc:
            return self._failed(
                source_path,
                digest,
                "pdf_open_failed",
                f"PDF could not be parsed ({type(exc).__name__}).",
                metadata=metadata,
                warnings=warnings,
                review_flags=document_flags | {"manual_review"},
            )

        if not pages:
            warnings.append(
                ParseWarning(
                    "no_pages",
                    "The PDF has no readable pages.",
                    review_flag="manual_review",
                )
            )
            document_flags.add("manual_review")

        return PDFParseResult(
            source_path=source_path,
            source_sha256=digest,
            parser_version=PARSER_VERSION,
            status="partial" if warnings else "complete",
            metadata=metadata,
            pages=tuple(pages),
            tables=tuple(tables),
            warnings=tuple(warnings),
            review_flags=tuple(sorted(document_flags)),
        )

    @staticmethod
    def _failed(
        path: Path,
        digest: str | None,
        code: str,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
        warnings: list[ParseWarning] | None = None,
        review_flags: set[str] | None = None,
    ) -> PDFParseResult:
        all_warnings = list(warnings or [])
        all_warnings.append(ParseWarning(code, message, review_flag="manual_review"))
        flags = set(review_flags or ())
        flags.add("manual_review")
        return PDFParseResult(
            source_path=path,
            source_sha256=digest,
            parser_version=PARSER_VERSION,
            status="failed",
            metadata=metadata or {},
            pages=(),
            tables=(),
            warnings=tuple(all_warnings),
            review_flags=tuple(sorted(flags)),
        )


def _extract_table(page, table, *, page_number: int, table_index: int) -> ExtractedTable:
    raw_rows = table.extract() or []
    normalized_rows = [
        ["" if cell is None else str(cell) for cell in row]
        for row in raw_rows
    ]
    column_count = max((len(row) for row in normalized_rows), default=0)
    bbox = tuple(float(value) for value in table.bbox) if table.bbox else None
    geometry_rows = tuple(getattr(table, "rows", ()))
    raw_cells = tuple(
        TableCell(
            row_index=row_index,
            column_index=column_index,
            text=row[column_index] if column_index < len(row) else "",
            bounding_box=_cell_bbox(geometry_rows, row_index, column_index),
        )
        for row_index, row in enumerate(normalized_rows)
        for column_index in range(column_count)
    )
    heading_context = _heading_context(page, table.bbox)
    caption_context = _caption_context(page, table.bbox)
    header_names = tuple(
        "" if name is None else str(name)
        for name in (getattr(getattr(table, "header", None), "names", ()) or ())
    )
    headers = header_names or (tuple(normalized_rows[0]) if normalized_rows else ())
    header_bbox_value = getattr(getattr(table, "header", None), "bbox", None)
    header_bbox = (
        tuple(float(value) for value in header_bbox_value)
        if header_bbox_value is not None
        else None
    )
    data_start = _first_data_row(normalized_rows, geometry_rows, header_bbox)
    header_rows = _header_row_indices(
        normalized_rows,
        geometry_rows,
        header_bbox,
        data_start,
        caption_context + "\n" + "\n".join(headers),
    )
    row_centers = _logical_row_centers(raw_cells, table.bbox)
    column_centers = _logical_column_centers(
        page,
        raw_cells,
        normalized_rows,
        geometry_rows,
        data_start,
        table.bbox,
        caption_context + "\n" + "\n".join(headers),
    )
    frequency_header_ranges, frequency_data_columns = _frequency_header_ranges(
        raw_cells,
        header_rows,
        column_centers,
    )
    logical_cells, expanded_cells, unmapped_cells = _logical_cells(
        page,
        raw_cells,
        normalized_rows,
        column_centers,
        row_centers,
        table.bbox,
        caption_context + "\n" + "\n".join(headers),
        frequency_header_ranges,
    )
    frequency_bands = _frequency_bands(
        raw_cells,
        logical_cells,
        header_rows,
        data_start,
        frequency_header_ranges,
        frequency_data_columns,
        column_centers,
        expanded_cells,
        caption_context + "\n" + "\n".join(headers),
    )
    frequency_labels = tuple(band.source_label for band in frequency_bands)
    header_text = _header_text(raw_cells, header_rows, headers)
    metric_context = "\n".join((caption_context, header_text))
    measurement_types, measurement_evidence = _measurement_types(metric_context)
    units = _extract_units(
        metric_context
        + "\n"
        + header_text
        + "\n"
        + "\n".join(cell.text for cell in raw_cells)
    )
    logical_column_count = len(column_centers)
    logical_row_count = max(
        len(normalized_rows),
        max(
            (
                cell.row_end if cell.row_end is not None else cell.row_index
                for cell in logical_cells
            ),
            default=-1,
        )
        + 1,
    )
    alignment_status, review_flags = _alignment_review(
        raw_cells,
        logical_cells,
        column_count,
        logical_column_count,
        unmapped_cells,
        expanded_cells,
    )
    if not measurement_types and _has_acoustic_header(metric_context):
        measurement_types = ("unspecified",)
        measurement_evidence = _measurement_source_phrases(
            caption_context,
            headers,
            raw_cells,
            header_rows,
        )
        review_flags = tuple(sorted(set(review_flags) | {"measurement_type_unspecified"}))

    return ExtractedTable(
        page_number=page_number,
        table_index=table_index,
        row_count=len(normalized_rows),
        column_count=column_count,
        cells=raw_cells,
        column_headers=headers,
        header_row_index=header_rows[0] if header_rows else None,
        heading_context=heading_context,
        frequency_band_labels=frequency_labels,
        units=units,
        bounding_box=bbox,
        logical_row_count=logical_row_count,
        logical_column_count=logical_column_count,
        logical_cells=logical_cells,
        header_row_indices=header_rows,
        frequency_bands=frequency_bands,
        measurement_types=measurement_types,
        measurement_type_evidence=measurement_evidence,
        alignment_status=alignment_status,
        review_flags=review_flags,
    )


def _cell_bbox(geometry_rows, row_index: int, column_index: int):
    if row_index >= len(geometry_rows):
        return None
    row_cells = getattr(geometry_rows[row_index], "cells", ())
    if column_index >= len(row_cells):
        return None
    box = row_cells[column_index]
    return tuple(float(value) for value in box) if box is not None else None


def _first_data_row(normalized_rows, geometry_rows, header_bbox) -> int | None:
    header_bottom = float(header_bbox[3]) if header_bbox is not None else None
    for row_index, row in enumerate(normalized_rows):
        if header_bottom is not None:
            row_boxes = [
                _cell_bbox(geometry_rows, row_index, column_index)
                for column_index, text in enumerate(row)
                if text.strip()
            ]
            row_top = min((box[1] for box in row_boxes if box), default=None)
            if row_top is not None and row_top < header_bottom - 0.5:
                continue
        nonempty = [text for text in row if text.strip()]
        numeric_cells = sum(bool(NUMBER_PATTERN.search(text)) for text in nonempty)
        if len(nonempty) >= 2 and numeric_cells >= 2:
            return row_index
    return None


def _header_row_indices(
    normalized_rows,
    geometry_rows,
    header_bbox,
    data_start: int | None,
    heading_context: str,
) -> tuple[int, ...]:
    stop = data_start if data_start is not None else len(normalized_rows)
    indices = [
        row_index
        for row_index, row in enumerate(normalized_rows[:stop])
        if any(text.strip() for text in row)
    ]
    if data_start is not None:
        return tuple(indices)

    header_bottom = float(header_bbox[3]) if header_bbox is not None else None
    for row_index, row in enumerate(normalized_rows):
        row_boxes = [
            _cell_bbox(geometry_rows, row_index, column_index)
            for column_index, text in enumerate(row)
            if text.strip()
        ]
        row_top = min((box[1] for box in row_boxes if box), default=None)
        inside_header = (
            header_bottom is not None
            and row_top is not None
            and row_top < header_bottom + 0.5
        )
        explicit_frequency = any(
            EXPLICIT_FREQUENCY_SUFFIX_PATTERN.search(match.group("label"))
            for text in row
            for match in FREQUENCY_VALUE_PATTERN.finditer(text)
        )
        if inside_header or (
            explicit_frequency
            and FREQUENCY_LABEL_PATTERN.search(heading_context + " " + " ".join(row))
        ):
            if any(text.strip() for text in row):
                indices.append(row_index)
    return tuple(sorted(set(indices)))


def _logical_column_centers(
    page,
    raw_cells: tuple[TableCell, ...],
    normalized_rows,
    geometry_rows,
    data_start: int | None,
    table_bbox,
    heading_context: str,
) -> tuple[float, ...]:
    if data_start is not None:
        candidate_rows = []
        for row_index in range(data_start, len(normalized_rows)):
            row_cells = [
                cell
                for cell in raw_cells
                if cell.row_index == row_index and cell.text.strip() and cell.bounding_box
            ]
            numeric_cells = sum(bool(NUMBER_PATTERN.search(cell.text)) for cell in row_cells)
            if len(row_cells) >= 3 and numeric_cells >= 2:
                candidate_rows.append(row_cells)
        if candidate_rows:
            selected = max(candidate_rows, key=lambda row: (len(row), len(row)))
            centers = [(cell.bounding_box[0] + cell.bounding_box[2]) / 2 for cell in selected]
            return _cluster_centers(centers)

    frequency_context = heading_context + "\n" + "\n".join(
        cell.text for cell in raw_cells if cell.text.strip()
    )
    if FREQUENCY_LABEL_PATTERN.search(frequency_context):
        explicit_words = []
        for cell in raw_cells:
            if not cell.bounding_box:
                continue
            for word in _words_in_box(page, cell.bounding_box):
                if _parse_frequency_label(word[4], allow_bare=False):
                    explicit_words.append((word[4], (word[0] + word[2]) / 2))
        band_centers = _cluster_centers([center for _, center in explicit_words])
        if len(band_centers) >= 2:
            left_edges = [
                cell.bounding_box
                for cell in raw_cells
                if cell.bounding_box
                and cell.text.strip()
                and cell.bounding_box[2] < band_centers[0]
                and not NUMBER_PATTERN.search(cell.text)
            ]
            label_centers = [
                (box[0] + box[2]) / 2
                for box in left_edges
                if box[2] < band_centers[0] - 4
            ]
            if label_centers:
                band_centers = _cluster_centers([*label_centers, *band_centers])
            return band_centers

    row_with_most_cells = max(
        (
            [
                cell
                for cell in raw_cells
                if cell.row_index == row_index and cell.text.strip() and cell.bounding_box
            ]
            for row_index in range(len(normalized_rows))
        ),
        key=len,
        default=[],
    )
    if row_with_most_cells:
        return _cluster_centers(
            [(cell.bounding_box[0] + cell.bounding_box[2]) / 2 for cell in row_with_most_cells]
        )
    if table_bbox:
        return (float(table_bbox[0]),)
    return ()


def _cluster_centers(values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    clusters: list[list[float]] = []
    for value in sorted(values):
        if clusters and value - sum(clusters[-1]) / len(clusters[-1]) <= 3.0:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return tuple(sum(cluster) / len(cluster) for cluster in clusters)


def _logical_row_centers(raw_cells: tuple[TableCell, ...], table_bbox) -> tuple[float, ...]:
    edges = []
    if table_bbox:
        edges.extend((float(table_bbox[1]), float(table_bbox[3])))
    for cell in raw_cells:
        if cell.bounding_box:
            edges.extend((cell.bounding_box[1], cell.bounding_box[3]))
    boundaries = _cluster_centers(edges)
    return tuple(
        (boundaries[index] + boundaries[index + 1]) / 2
        for index in range(len(boundaries) - 1)
        if boundaries[index + 1] - boundaries[index] > 1.0
    )


def _cell_rows(
    bbox: tuple[float, float, float, float] | None,
    centers: tuple[float, ...],
    fallback: int,
) -> tuple[int, int]:
    if bbox is None or not centers:
        return fallback, fallback
    indices = [
        index
        for index, center in enumerate(centers)
        if bbox[1] - 1.0 <= center <= bbox[3] + 1.0
    ]
    if not indices:
        return fallback, fallback
    return indices[0], indices[-1]


def _cell_columns(
    bounding_box: tuple[float, float, float, float] | None,
    centers: tuple[float, ...],
) -> tuple[int | None, int | None]:
    if not centers:
        return None, None
    if bounding_box is None:
        return None, None
    left, _, right, _ = bounding_box
    covered = [
        index
        for index, center in enumerate(centers)
        if left - 1.0 <= center <= right + 1.0
    ]
    if covered:
        return covered[0], covered[-1]
    center = (left + right) / 2
    nearest = min(range(len(centers)), key=lambda index: abs(centers[index] - center))
    if abs(centers[nearest] - center) <= max(6.0, (right - left) / 2 + 2.0):
        return nearest, nearest
    return None, None


def _frequency_header_ranges(
    raw_cells: tuple[TableCell, ...],
    header_rows: tuple[int, ...],
    centers: tuple[float, ...],
) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
    ranges = []
    data_columns = []
    for cell in raw_cells:
        if cell.row_index not in header_rows or not FREQUENCY_LABEL_PATTERN.search(cell.text):
            continue
        start, end = _cell_columns(cell.bounding_box, centers)
        if start is not None and end is not None:
            ranges.append((start, end))
            if start == end and re.search(r"\bfrequency\s+bands?\b", cell.text, re.I):
                data_columns.append(start)
    return tuple(ranges), tuple(sorted(set(data_columns)))


def _logical_cells(
    page,
    raw_cells: tuple[TableCell, ...],
    normalized_rows,
    centers: tuple[float, ...],
    row_centers: tuple[float, ...],
    table_bbox,
    heading_context: str,
    frequency_ranges: tuple[tuple[int, int], ...],
) -> tuple[tuple[LogicalTableCell, ...], tuple[LogicalTableCell, ...], int]:
    output: list[LogicalTableCell] = []
    expanded: list[LogicalTableCell] = []
    expanded_sources: set[tuple[int, int]] = set()
    context = heading_context + "\n" + " ".join(
        cell.text for cell in raw_cells if cell.row_index == 0
    )

    for cell in raw_cells:
        if not cell.text.strip():
            continue
        split_cells = _split_frequency_value_cell(
            page,
            cell,
            centers,
            frequency_ranges,
            context,
        )
        if split_cells:
            expanded.extend(split_cells)
            expanded_sources.add((cell.row_index, cell.column_index))

    for cell in raw_cells:
        if not cell.text.strip() or (cell.row_index, cell.column_index) in expanded_sources:
            continue
        start, end = _cell_columns(cell.bounding_box, centers)
        row_start, row_end = _cell_rows(cell.bounding_box, row_centers, cell.row_index)
        output.append(
            LogicalTableCell(
                row_index=row_start,
                column_start=start,
                column_end=end,
                text=cell.text,
                source_row_index=cell.row_index,
                source_column_index=cell.column_index,
                bounding_box=cell.bounding_box,
                row_start=row_start,
                row_end=row_end,
            )
        )
    output.extend(expanded)
    unmapped = sum(
        1
        for cell in output
        if cell.text.strip() and cell.column_start is None
    )
    output.sort(key=lambda cell: (cell.row_index, cell.column_start or -1, cell.source_column_index))
    return tuple(output), tuple(expanded), unmapped


def _split_frequency_value_cell(
    page,
    cell: TableCell,
    centers: tuple[float, ...],
    frequency_ranges: tuple[tuple[int, int], ...],
    context: str,
) -> tuple[LogicalTableCell, ...]:
    if not cell.bounding_box or "\n" not in cell.text or not centers:
        return ()
    if not FREQUENCY_LABEL_PATTERN.search(context):
        return ()
    words = _words_in_box(page, cell.bounding_box)
    lines = _group_word_lines(words)
    frequency_lines = []
    value_lines = []
    for line in lines:
        frequencies = [
            word
            for word in line
            if _parse_frequency_label(word[4], allow_bare=False) is not None
        ]
        values = [word for word in line if re.fullmatch(r"-?\d+(?:\.\d+)?", word[4])]
        if len(frequencies) >= 2:
            frequency_lines.append(frequencies)
        if len(values) >= 2:
            value_lines.append(values)
    for frequency_words in frequency_lines:
        for value_words in value_lines:
            if len(frequency_words) != len(value_words):
                continue
            frequency_words.sort(key=lambda word: word[0])
            value_words.sort(key=lambda word: word[0])
            if any(
                abs((left[0] + left[2]) / 2 - (right[0] + right[2]) / 2) > 4.0
                for left, right in zip(frequency_words, value_words)
            ):
                continue
            frequency_row = cell.row_index
            value_row = cell.row_index + 1
            output = []
            for word in frequency_words:
                label = word[4]
                center_hz = _parse_frequency_label(label, allow_bare=False)
                logical_column = _nearest_column((word[0] + word[2]) / 2, centers)
                if center_hz is None or logical_column is None:
                    return ()
                output.append(
                    LogicalTableCell(
                        row_index=frequency_row,
                        column_start=logical_column,
                        column_end=logical_column,
                        text=label,
                        source_row_index=cell.row_index,
                        source_column_index=cell.column_index,
                        bounding_box=(word[0], word[1], word[2], word[3]),
                        row_start=frequency_row,
                        row_end=frequency_row,
                    )
                )
            for word in value_words:
                logical_column = _nearest_column((word[0] + word[2]) / 2, centers)
                if logical_column is None:
                    return ()
                output.append(
                    LogicalTableCell(
                        row_index=value_row,
                        column_start=logical_column,
                        column_end=logical_column,
                        text=word[4],
                        source_row_index=cell.row_index,
                        source_column_index=cell.column_index,
                        bounding_box=(word[0], word[1], word[2], word[3]),
                        row_start=value_row,
                        row_end=value_row,
                    )
                )
            return tuple(output)
    return ()


def _frequency_bands(
    raw_cells: tuple[TableCell, ...],
    logical_cells: tuple[LogicalTableCell, ...],
    header_rows: tuple[int, ...],
    data_start: int | None,
    frequency_ranges: tuple[tuple[int, int], ...],
    frequency_data_columns: tuple[int, ...],
    centers: tuple[float, ...],
    expanded_cells: tuple[LogicalTableCell, ...],
    heading_context: str,
) -> tuple[FrequencyBand, ...]:
    bands: dict[tuple[int, int], FrequencyBand] = {}
    header_context = heading_context + "\n" + "\n".join(
        cell.text for cell in raw_cells if cell.row_index in header_rows
    )
    frequency_context = bool(FREQUENCY_LABEL_PATTERN.search(header_context))
    expanded_frequency_sources = {
        (cell.source_row_index, cell.source_column_index)
        for cell in expanded_cells
        if _parse_frequency_label(cell.text, allow_bare=False) is not None
    }

    for cell in raw_cells:
        if not cell.text.strip():
            continue
        start, end = _cell_columns(cell.bounding_box, centers)
        if start is None or end is None:
            continue
        in_frequency_column = any(
            start <= range_end and end >= range_start
            for range_start, range_end in frequency_ranges
        )
        is_header_cell = cell.row_index in header_rows and in_frequency_column
        is_frequency_data_cell = (
            data_start is not None
            and cell.row_index >= data_start
            and start == end
            and start in frequency_data_columns
            and (
                bool(re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", cell.text))
                or _parse_frequency_label(cell.text, allow_bare=False) is not None
            )
        )
        if not (
            (is_header_cell and frequency_context)
            or is_frequency_data_cell
            or (cell.row_index, cell.column_index) in expanded_frequency_sources
        ):
            continue
        for match in FREQUENCY_VALUE_PATTERN.finditer(cell.text):
            label = match.group("label").strip()
            is_explicit = bool(EXPLICIT_FREQUENCY_SUFFIX_PATTERN.search(label))
            bare_cell = bool(re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", cell.text))
            if not is_explicit and not (
                bare_cell
                and frequency_context
                and (is_header_cell or is_frequency_data_cell)
            ):
                continue
            center_hz = _parse_frequency_label(
                label,
                allow_bare=frequency_context and (is_header_cell or is_frequency_data_cell),
            )
            if center_hz is None:
                continue
            logical_column = _nearest_column_for_cell(cell, centers)
            if logical_column is None:
                continue
            key = (logical_column, cell.row_index)
            current = bands.get(key)
            if current is None:
                bands[key] = FrequencyBand(label, center_hz, logical_column, cell.row_index)
            elif current.center_hz != center_hz:
                continue

    for cell in expanded_cells:
        center_hz = _parse_frequency_label(cell.text, allow_bare=False)
        if center_hz is None or cell.column_start is None:
            continue
        key = (cell.column_start, cell.row_index)
        bands[key] = FrequencyBand(
            cell.text,
            center_hz,
            cell.column_start,
            cell.source_row_index,
        )
    return tuple(bands[key] for key in sorted(bands))


def _parse_frequency_label(value: str, *, allow_bare: bool) -> float | None:
    match = FREQUENCY_VALUE_PATTERN.fullmatch(value.strip())
    if not match:
        return None
    label = match.group("label")
    suffix_match = re.search(r"(kHz|Hz|k)$", label, re.IGNORECASE)
    if not suffix_match and not allow_bare:
        return None
    number_text = re.match(r"\d+(?:\.\d+)?", label)
    if not number_text:
        return None
    frequency = float(number_text.group(0))
    suffix = suffix_match.group(1).lower() if suffix_match else "hz"
    if suffix in {"khz", "k"}:
        frequency *= 1000
    if not math.isfinite(frequency) or frequency <= 0 or frequency > 200000:
        return None
    return frequency


def _nearest_column(x: float, centers: tuple[float, ...]) -> int | None:
    if not centers:
        return None
    index = min(range(len(centers)), key=lambda item: abs(centers[item] - x))
    return index if abs(centers[index] - x) <= 8.0 else None


def _nearest_column_for_cell(
    cell: TableCell,
    centers: tuple[float, ...],
) -> int | None:
    if cell.bounding_box is None:
        return None
    start, end = _cell_columns(cell.bounding_box, centers)
    if start is None or end is None:
        return None
    if start == end:
        return start
    center = (cell.bounding_box[0] + cell.bounding_box[2]) / 2
    return _nearest_column(center, centers)


def _words_in_box(page, bbox):
    if bbox is None:
        return ()
    return page.get_text("words", clip=pymupdf.Rect(*bbox), sort=True)


def _group_word_lines(words):
    lines: list[list[tuple]] = []
    for word in sorted(words, key=lambda item: (float(item[1]), float(item[0]))):
        center_y = (float(word[1]) + float(word[3])) / 2
        line = next(
            (
                existing
                for existing in lines
                if abs(
                    center_y
                    - sum((float(item[1]) + float(item[3])) / 2 for item in existing)
                    / len(existing)
                )
                <= 2.5
            ),
            None,
        )
        if line is None:
            lines.append([word])
        else:
            line.append(word)
    for line in lines:
        line.sort(key=lambda item: float(item[0]))
    return lines


def _header_text(
    raw_cells: tuple[TableCell, ...],
    header_rows: tuple[int, ...],
    headers: tuple[str, ...],
) -> str:
    return "\n".join(
        [
            *headers,
            *(cell.text for cell in raw_cells if cell.row_index in header_rows and cell.text.strip()),
        ]
    )


def _measurement_types(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    types = []
    evidence = []
    for metric_type, pattern in MEASUREMENT_TYPE_PATTERNS:
        for match in pattern.finditer(text):
            if metric_type not in types:
                types.append(metric_type)
            phrase = match.group(0).strip()
            if phrase not in evidence:
                evidence.append(phrase)
    return tuple(types), tuple(evidence)


def _has_acoustic_header(text: str) -> bool:
    return bool(ACOUSTIC_SIGNAL_PATTERN.search(text) or re.search(r"\b(?:SRI|SPL)\b", text, re.I))


def _source_phrases(text: str) -> tuple[str, ...]:
    normalized = " ".join(text.split())
    phrases = []
    for match in ACOUSTIC_SIGNAL_PATTERN.finditer(normalized):
        start = max(0, match.start() - 90)
        end = min(len(normalized), match.end() + 90)
        phrases.append(normalized[start:end])
    return tuple(dict.fromkeys(phrases))


def _measurement_source_phrases(
    caption_context: str,
    headers: tuple[str, ...],
    raw_cells: tuple[TableCell, ...],
    header_rows: tuple[int, ...],
) -> tuple[str, ...]:
    candidates = [
        caption_context,
        " ".join(headers),
        *(
            " ".join(
                cell.text
                for cell in raw_cells
                if cell.row_index == row_index and cell.text.strip()
            )
            for row_index in header_rows
        ),
    ]
    return tuple(
        dict.fromkeys(
            " ".join(candidate.split())
            for candidate in candidates
            if candidate.strip()
            and _has_acoustic_header(candidate)
            and MEASUREMENT_CONTEXT_PATTERN.search(candidate)
        )
    )


def _extract_units(text: str) -> tuple[str, ...]:
    units = []
    for match in UNIT_PATTERN.finditer(text):
        unit = match.group(0).strip()
        if unit not in units:
            units.append(unit)
    for match in SOUND_POWER_REFERENCE_PATTERN.finditer(text):
        unit = " ".join(match.group(0).split())
        if unit not in units:
            units.append(unit)
    return tuple(units)


def _caption_context(page, bbox) -> str:
    if not bbox:
        return ""
    top = float(bbox[1])
    captions = [
        block
        for block in page.get_text("blocks")
        if len(block) > 4
        and str(block[4]).strip()
        and float(block[3]) <= top
        and top - float(block[3]) <= 60
        and re.search(r"\btable\s+\d+(?:[-–.]\d+)*\b", str(block[4]), re.I)
    ]
    if not captions:
        return ""
    closest = max(captions, key=lambda block: float(block[3]))
    return " ".join(str(closest[4]).split())


def _alignment_review(
    raw_cells: tuple[TableCell, ...],
    logical_cells: tuple[LogicalTableCell, ...],
    raw_column_count: int,
    logical_column_count: int,
    unmapped_cells: int,
    expanded_cells: tuple[LogicalTableCell, ...],
) -> tuple[str, tuple[str, ...]]:
    flags = set()
    if unmapped_cells:
        flags.add("ambiguous_table_alignment")
        status = "ambiguous"
    elif logical_column_count == 0:
        flags.add("ambiguous_table_alignment")
        status = "ambiguous"
    elif raw_column_count != logical_column_count or expanded_cells:
        status = "reconstructed"
    else:
        status = "aligned"
    if status == "ambiguous":
        flags.add("table_human_review")
    return status, tuple(sorted(flags))


def _table_review_message(flag: str) -> str:
    if flag == "ambiguous_table_alignment":
        return "Table cells could not be mapped confidently to logical columns; raw cells were preserved."
    return "Table requires human review."


def _heading_context(page, bbox) -> str:
    if not bbox:
        return ""
    top = float(bbox[1])
    blocks = page.get_text("blocks")
    preceding = [
        block
        for block in blocks
        if len(block) > 4 and str(block[4]).strip() and float(block[3]) <= top
    ]
    preceding.sort(key=lambda block: (float(block[3]), float(block[1])), reverse=True)
    selected: list[str] = []
    for block in preceding:
        if top - float(block[3]) > 120:
            break
        text = " ".join(str(block[4]).split())
        if text:
            selected.append(text)
        if len(selected) == 3:
            break
    return "\n".join(reversed(selected))


def _normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        if isinstance(value, (str, int, float, bool)) or value is None
        else str(value)
        for key, value in metadata.items()
    }


def _unique_matches(pattern: re.Pattern[str], text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(match.group(0).strip() for match in pattern.finditer(text)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
