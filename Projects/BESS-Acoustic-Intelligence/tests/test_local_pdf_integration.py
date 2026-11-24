import os
import sqlite3
from pathlib import Path

import pytest

from bai.config import PROJECT_ROOT
from bai.documents.parser import PDFParser


REFERENCE_TABLES = (
    {
        "document_title": "EFSEC",
        "source_table": "Table 7",
        "page_number": 19,
        "table_index": 0,
        "logical_shape": (6, 11),
        "measurement_type": "sound_power_level",
        "anchor_cell": (0, 0, "Sound Source"),
        "unit_markers": ("dBL", "dBA"),
        "frequency_limits_hz": (31.5, 8000),
    },
    {
        "document_title": "AKRF",
        "source_table": "Table 2",
        "page_number": 7,
        "table_index": 0,
        "logical_shape": (5, 11),
        "measurement_type": "sound_power_level",
        "anchor_cell": (0, 0, "Noise Source"),
        "unit_markers": ("Hertz", "dBA"),
        "frequency_limits_hz": (63, 8000),
    },
    {
        "document_title": "CT.gov",
        "source_table": "Table 5-2",
        "page_number": 15,
        "table_index": 0,
        "logical_shape": (7, 12),
        "measurement_type": "sound_pressure_level",
        "anchor_cell": (0, 0, "Measurement Location ID"),
        "unit_markers": ("Hz", "dB", "dBA"),
        "frequency_limits_hz": (31.5, 16000),
    },
    {
        "document_title": "Preliminary Noise Assessment",
        "source_table": "Table 4-3",
        "page_number": 21,
        "table_index": 0,
        "logical_shape": (10, 5),
        "measurement_type": "unspecified",
        "anchor_cell": (1, 0, "Sensitive Receptor"),
        "unit_markers": ("m", "dB(A)"),
        "measurement_evidence_contains": ("Predicted", "Leq"),
    },
    {
        "document_title": "BS8233",
        "source_table": "Table 9",
        "page_number": 20,
        "table_index": 2,
        "logical_shape": (3, 7),
        "measurement_type": "sound_reduction_index",
        "anchor_cell": (0, 0, "Element"),
        "unit_markers": ("dB", "Hz", "kHz"),
        "frequency_limits_hz": (125, 4000),
    },
)


@pytest.mark.skipif(
    os.getenv("BAI_RUN_PDF_INTEGRATION") != "1",
    reason="set BAI_RUN_PDF_INTEGRATION=1 to parse local, ignored research PDFs",
)
def test_five_manually_verified_tables_retain_source_relationships():
    database_path = PROJECT_ROOT / "data" / "processed" / "bai.sqlite3"
    assert database_path.is_file(), "The local collection database is unavailable."
    with sqlite3.connect(database_path) as connection:
        documents = connection.execute(
            """
            SELECT title, local_path
            FROM documents
            WHERE download_status = 'downloaded'
            """
        ).fetchall()
    document_paths = {}
    for title, local_path in documents:
        matching = [
            reference
            for reference in REFERENCE_TABLES
            if reference["document_title"].casefold() in (title or "").casefold()
        ]
        if matching and local_path:
            path = Path(local_path)
            document_paths[matching[0]["document_title"]] = (
                path if path.is_absolute() else PROJECT_ROOT / path
            )

    parser = PDFParser()
    for reference in REFERENCE_TABLES:
        path = document_paths.get(reference["document_title"])
        assert path and path.is_file(), (
            f"Missing local source PDF for {reference['document_title']}."
        )
        result = parser.parse(path)
        table = next(
            (
                item
                for item in result.tables
                if item.page_number == reference["page_number"]
                and item.table_index == reference["table_index"]
            ),
            None,
        )
        assert table is not None, (
            f"{reference['source_table']} not found for {reference['document_title']}."
        )
        assert (table.logical_row_count, table.logical_column_count) == reference[
            "logical_shape"
        ]
        assert reference["measurement_type"] in table.measurement_types
        assert table.logical_cells
        assert table.cells
        assert table.alignment_status in {"aligned", "reconstructed"}

        anchor_row, anchor_column, anchor_text = reference["anchor_cell"]
        anchor = next(
            (
                cell.text
                for cell in table.logical_cells
                if cell.row_index == anchor_row
                and cell.column_start == anchor_column
            ),
            "",
        )
        assert " ".join(anchor.split()) == anchor_text
        assert set(reference["unit_markers"]).issubset(table.units)

        frequency_limits = reference.get("frequency_limits_hz")
        if frequency_limits:
            centers = {band.center_hz for band in table.frequency_bands}
            assert frequency_limits[0] in centers
            assert frequency_limits[1] in centers

        if reference["measurement_type"] == "unspecified":
            assert "measurement_type_unspecified" in table.review_flags
        for marker in reference.get("measurement_evidence_contains", ()):
            assert marker.casefold() in " ".join(
                table.measurement_type_evidence
            ).casefold()
