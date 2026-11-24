import os
import re
import sqlite3
from pathlib import Path

import pytest
import pymupdf

from bai.config import PROJECT_ROOT
from bai.database import Database
from bai.documents.parser import PDFParser
from bai.engineering.benchmark_store import BenchmarkStore
from bai.engineering.semantic_benchmark import SemanticBenchmarkGenerator


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


@pytest.mark.skipif(
    os.getenv("BAI_RUN_PDF_INTEGRATION") != "1",
    reason="set BAI_RUN_PDF_INTEGRATION=1 to compare candidates with local research PDFs",
)
def test_ten_semantic_regressions_match_original_pdf_pages(tmp_path):
    database_path = PROJECT_ROOT / "data" / "processed" / "bai.sqlite3"
    assert database_path.is_file(), "The local collection database is unavailable."
    database = Database(database_path)
    generator = SemanticBenchmarkGenerator(
        database,
        BenchmarkStore(tmp_path / "unused-benchmark.sqlite3"),
    )
    grouped, _ = generator.collect_candidates()
    candidates = [candidate for group in grouped.values() for candidate in group]

    cases = (
        {
            "title": "Heywood BESS",
            "page": 16,
            "value": "83",
            "entity": "Inverter",
            "entity_role": "equipment",
            "table": "Table 7",
            "record_kind": "equipment_sound_emission",
            "metric_type": "sound_power",
            "value_basis": "unknown",
            "relevance": "bess_specific",
            "aggregation": "per_unit",
            "unit": "dBA",
            "terms": ("Table 7", "Sound Power", "Inverter"),
        },
        {
            "title": "Heywood BESS",
            "page": 16,
            "value": "88",
            "entity": "High Voltage Transformer",
            "entity_role": "equipment",
            "table": "Table 7",
            "record_kind": "equipment_sound_emission",
            "metric_type": "sound_power",
            "value_basis": "unknown",
            "relevance": "bess_specific",
            "aggregation": "per_unit",
            "unit": "dBA",
            "terms": ("Table 7", "Sound Power", "Transformer"),
        },
        {
            "title": "APPENDIX F",
            "page": 11,
            "value": "140",
            "entity": "Jet aircraft at 75 feet",
            "entity_role": "environment_description",
            "table": "Table 2-1",
            "record_kind": "general_acoustic_reference_example",
            "metric_type": "sound_pressure",
            "value_basis": "unknown",
            "relevance": "general_reference",
            "unit": "dBA",
            "terms": ("Table 2-1", "Jet aircraft at 75 feet"),
        },
        {
            "title": "Preliminary Noise Assessment",
            "page": 14,
            "value": "66",
            "entity": "Monitoring Location 3",
            "entity_role": "monitoring_location",
            "table": "TABLE 2-3",
            "record_kind": "ambient_environmental_measurement",
            "metric_type": "unknown",
            "value_basis": "measured",
            "relevance": "bess_specific",
            "unit": "dB(A)",
            "descriptor": "Aeq",
            "terms": ("TABLE 2-3", "Monitoring Location", "3", "LAeq"),
        },
        {
            "title": "Preliminary Noise Assessment",
            "page": 14,
            "value": "46",
            "entity": "Monitoring Location 3",
            "entity_role": "monitoring_location",
            "table": "TABLE 2-3",
            "record_kind": "ambient_environmental_measurement",
            "metric_type": "unknown",
            "value_basis": "measured",
            "relevance": "bess_specific",
            "unit": "dB(A)",
            "descriptor": "AF90",
            "terms": ("TABLE 2-3", "Monitoring Location", "3", "LAF90"),
        },
        {
            "title": "EFSEC",
            "page": 10,
            "value": "35",
            "entity": "Typical wilderness area",
            "entity_role": "environment_description",
            "table": "Table 1",
            "record_kind": "general_acoustic_reference_example",
            "metric_type": "sound_pressure",
            "value_basis": "unknown",
            "relevance": "general_reference",
            "unit": "dBA",
            "terms": ("Table 1", "Typical wilderness area"),
        },
        {
            "title": "AKRF",
            "page": 5,
            "value": "85",
            "entity": None,
            "entity_role": "unknown",
            "table": None,
            "record_kind": "regulatory_noise_criterion",
            "metric_type": "sound_pressure",
            "value_basis": "unknown",
            "relevance": "uncertain",
            "unit": "Decibels",
            "frequency_hz": "31.5",
            "terms": ("Daytime Octave Band Sound Pressure Level Limits", "31.5"),
        },
        {
            "title": "CT.gov",
            "page": 10,
            "value": "16",
            "entity": None,
            "entity_role": "unknown",
            "table": "Table 4-2",
            "record_kind": "regulatory_noise_criterion",
            "metric_type": "unknown",
            "value_basis": "unknown",
            "relevance": "uncertain",
            "unit": "dB",
            "frequency_hz": "100",
            "terms": ("Determination of a Prominent Discrete Tone", "100"),
        },
        {
            "title": "Heywood BESS",
            "page": 17,
            "value": "33",
            "entity": "R2",
            "entity_role": "receiver",
            "table": "Table 8",
            "record_kind": "predicted_receptor_noise_level",
            "metric_type": "unknown",
            "value_basis": "predicted",
            "relevance": "bess_specific",
            "unit": "dBA",
            "terms": ("Table 8", "Predicted Noise", "Receiver", "R2"),
        },
        {
            "title": "BS8233",
            "page": 21,
            "value": "≤10",
            "entity": None,
            "entity_contains": "glass",
            "entity_role": "building_configuration",
            "table": "Table 10",
            "record_kind": "building_sound_insulation_performance",
            "metric_type": "unknown",
            "value_basis": "unknown",
            "relevance": "cross_industry",
            "unit": "dB",
            "comparator": "lte",
            "upper_bound": "10",
            "terms": ("Table 10", "Rw+Ctr"),
        },
    )

    with sqlite3.connect(database_path) as connection:
        pdf_paths = dict(
            connection.execute("SELECT id, local_path FROM documents")
        )

    for case in cases:
        matches = [
            candidate
            for candidate in candidates
            if case["title"].casefold() in candidate.source.title.casefold()
            and candidate.page_number == case["page"]
            and (
                (
                    candidate.measurement.comparator == case["comparator"]
                    and str(candidate.measurement.upper_bound)
                    == case["upper_bound"]
                )
                if "comparator" in case
                else candidate.measurement.source_value_text == case["value"]
            )
            and (
                case["entity"] is None
                or candidate.source_entity_label == case["entity"]
            )
            and (
                "entity_contains" not in case
                or case["entity_contains"].casefold()
                in (candidate.source_entity_label or "").casefold()
            )
            and candidate.source_entity_role == case["entity_role"]
            and candidate.source_table_reference == case["table"]
        ]
        assert len(matches) == 1, case
        candidate = matches[0]
        measurement = candidate.measurement
        assert candidate.record_kind == case["record_kind"]
        assert candidate.source_entity_role == case["entity_role"]
        assert measurement.metric_type == case["metric_type"]
        assert measurement.value_basis == case["value_basis"]
        assert candidate.bess_relevance == case["relevance"]
        assert measurement.units == case["unit"]
        if "aggregation" in case:
            assert measurement.aggregation_basis == case["aggregation"]
        if "comparator" in case:
            assert measurement.comparator == case["comparator"]
            assert str(measurement.upper_bound) == case["upper_bound"]
            assert measurement.source_value_text.replace(" ", "") == "\u226410"
        if "descriptor" in case:
            assert case["descriptor"].casefold() in (
                measurement.metric_descriptor or ""
            ).casefold()
        if "frequency_hz" in case:
            assert measurement.frequency_band is not None
            assert str(measurement.frequency_band.center_hz) == case["frequency_hz"]

        pdf_path = Path(pdf_paths[candidate.source.document_id])
        if not pdf_path.is_absolute():
            pdf_path = PROJECT_ROOT / pdf_path
        with pymupdf.open(pdf_path) as document:
            page_text = " ".join(
                document[candidate.page_number - 1].get_text().split()
            ).casefold()
        if "comparator" in case:
            assert re.search(r"≤\s*10", page_text)
        else:
            assert case["value"].casefold() in page_text
        for term in case["terms"]:
            assert " ".join(term.split()).casefold() in page_text
        if measurement.frequency_band is not None:
            assert (
                measurement.frequency_band.source_label.casefold() in page_text
            )
