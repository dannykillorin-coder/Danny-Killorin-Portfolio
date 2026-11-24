import hashlib
import json
import sqlite3
from pathlib import Path

import pymupdf

from bai.database import Database
from bai.documents.classification import classify_pages
from bai.documents.models import ExtractedPage
from bai.documents.parser import PDFParser
from bai.documents.pipeline import DocumentIntelligence
from bai.documents.store import DocumentStore
from bai.models import Candidate, DownloadResult


def _make_text_pdf(path: Path, text: str = "BESS acoustic impact assessment") -> Path:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.set_metadata({"title": "BAI deterministic test"})
    document.save(path)
    document.close()
    return path


def _make_table_pdf(
    path: Path,
    *,
    frequency_header: str = "Frequency band (Hz)",
    bands: tuple[str, str] = ("63", "125"),
) -> Path:
    document = pymupdf.open()
    page = document.new_page(width=400, height=320)
    page.insert_text((40, 55), "Octave-band sound pressure measurements dB(A)")
    x_positions = (40, 200, 360)
    y_positions = (90, 130, 170, 210)
    for x in x_positions:
        page.draw_line((x, y_positions[0]), (x, y_positions[-1]))
    for y in y_positions:
        page.draw_line((x_positions[0], y), (x_positions[-1], y))
    values = (
            (frequency_header, "Sound pressure dB(A)"),
            (bands[0], "70 dB(A)"),
            (bands[1], "65 dB(A)"),
    )
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            page.insert_text(
                (x_positions[column_index] + 5, y_positions[row_index] + 25),
                value,
                fontsize=9,
            )
    document.set_metadata({"title": "Table test"})
    document.save(path)
    document.close()
    return path


def _make_numeric_measurement_pdf(path: Path) -> Path:
    document = pymupdf.open()
    page = document.new_page(width=400, height=320)
    page.insert_text((40, 55), "Table 1: Octave-band equipment noise levels")
    x_positions = (40, 200, 360)
    y_positions = (90, 130, 170, 210)
    for x in x_positions:
        page.draw_line((x, y_positions[0]), (x, y_positions[-1]))
    for y in y_positions:
        page.draw_line((x_positions[0], y), (x_positions[-1], y))
    values = (
        ("Equipment", "Level dB(A)"),
        ("Inverter A", "63"),
        ("Inverter B", "125"),
    )
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            page.insert_text(
                (x_positions[column_index] + 5, y_positions[row_index] + 25),
                value,
                fontsize=9,
            )
    document.save(path)
    document.close()
    return path


def _make_image_only_pdf(path: Path) -> Path:
    document = pymupdf.open()
    page = document.new_page()
    pixmap = pymupdf.Pixmap(
        pymupdf.csRGB,
        pymupdf.IRect(0, 0, 50, 50),
        0,
    )
    pixmap.clear_with(240)
    page.insert_image(pymupdf.Rect(50, 50, 300, 300), stream=pixmap.tobytes("png"))
    document.save(path)
    document.close()
    return path


def _record_download(
    database: Database,
    pdf_path: Path,
    title: str = "Deterministic test document",
) -> int:
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    document_id, _ = database.add_candidate(
        Candidate(
            url="https://example.org/test.pdf",
            title=title,
            query="test",
            provider="manual",
        )
    )
    database.save_download(
        document_id,
        DownloadResult(
            status="downloaded",
            local_path=str(pdf_path),
            file_size=pdf_path.stat().st_size,
            sha256=digest,
        ),
    )
    return document_id


def test_parser_extracts_pages_metadata_and_acoustic_signals(tmp_path):
    pdf_path = _make_text_pdf(
        tmp_path / "assessment.pdf",
        "BESS acoustic impact assessment; sound pressure level is 65 dB(A).",
    )

    result = PDFParser().parse(pdf_path)

    assert result.status == "complete"
    assert result.source_sha256 == hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    assert result.metadata["title"] == "BAI deterministic test"
    assert result.pages[0].page_number == 1
    assert "65 dB(A)" in result.pages[0].text
    assert "dB(A)" in result.pages[0].acoustic_signals
    assert not result.review_flags


def test_parser_preserves_table_cell_positions_headers_bands_units_and_page(tmp_path):
    result = PDFParser().parse(_make_table_pdf(tmp_path / "table.pdf"))

    assert len(result.tables) == 1
    table = result.tables[0]
    assert table.page_number == 1
    assert table.row_count == 3
    assert table.column_count == 2
    assert table.header_row_index == 0
    assert table.column_headers == ("Frequency band (Hz)", "Sound pressure dB(A)")
    assert {"63", "125"}.issubset(table.frequency_band_labels)
    assert "Hz" in table.units
    assert "dB(A)" in table.units
    assert any(
        cell.row_index == 2 and cell.column_index == 1 and cell.text == "65 dB(A)"
        for cell in table.cells
    )
    assert "Octave-band sound pressure measurements" in table.heading_context


def test_parser_does_not_treat_numeric_noise_levels_as_frequency_bands(tmp_path):
    result = PDFParser().parse(
        _make_numeric_measurement_pdf(tmp_path / "numeric-levels.pdf")
    )

    assert len(result.tables) == 1
    table = result.tables[0]
    assert table.frequency_bands == ()
    assert table.frequency_band_labels == ()


def test_parser_normalizes_khz_frequency_bands_without_losing_source_labels(tmp_path):
    result = PDFParser().parse(
        _make_table_pdf(
            tmp_path / "kilohertz.pdf",
            frequency_header="Frequency band (kHz)",
            bands=("0.063 kHz", "0.125 kHz"),
        )
    )

    table = result.tables[0]
    assert {band.center_hz for band in table.frequency_bands} == {63, 125}
    assert table.frequency_band_labels == ("0.063 kHz", "0.125 kHz")
    assert "kHz" in table.units


def test_parser_flags_image_only_pages_without_calling_content_absent(tmp_path):
    result = PDFParser().parse(_make_image_only_pdf(tmp_path / "scan.pdf"))

    assert result.status == "partial"
    assert result.pages[0].text_status == "image_heavy"
    assert "ocr_review" in result.pages[0].review_flags
    assert result.warnings[0].code == "image_heavy_page"
    assert "does not establish" not in result.warnings[0].message
    assert "may be unavailable" in result.warnings[0].message


def test_parser_handles_malformed_pdf_with_review_warning(tmp_path):
    malformed = tmp_path / "broken.pdf"
    malformed.write_bytes(b"not a PDF")

    result = PDFParser().parse(malformed)

    assert result.status == "failed"
    assert result.warnings[0].code == "pdf_open_failed"
    assert "manual_review" in result.review_flags


def test_parser_flags_frequency_band_content_when_no_table_was_reconstructed(tmp_path):
    pdf_path = _make_text_pdf(
        tmp_path / "table-like.pdf",
        "Table 2: Octave-band sound pressure levels are reported by frequency band.",
    )

    result = PDFParser().parse(pdf_path)

    assert result.tables == ()
    assert any(item.code == "candidate_table_not_detected" for item in result.warnings)
    assert "table_review" in result.review_flags


def test_classification_separates_direct_bess_and_cross_industry_evidence():
    bess_page = ExtractedPage(
        page_number=2,
        text="Battery energy storage noise impact assessment with sound pressure levels.",
        text_status="extracted",
        text_character_count=74,
        image_count=0,
        acoustic_signals=("sound pressure",),
        review_flags=(),
    )
    wind_page = ExtractedPage(
        page_number=1,
        text="Wind turbine noise attenuation and octave-band measurements.",
        text_status="extracted",
        text_character_count=61,
        image_count=0,
        acoustic_signals=("noise", "octave-band"),
        review_flags=(),
    )

    bess_results = classify_pages(
        [bess_page],
        document_title="Battery energy storage acoustic assessment",
    )
    wind_results = classify_pages([wind_page])
    bess_primary = next(item for item in bess_results if item.is_primary)
    wind_primary = next(item for item in wind_results if item.is_primary)

    assert bess_primary.category == "bess_acoustic_impact_assessment"
    assert bess_primary.direct_bess_evidence is True
    assert bess_primary.cross_industry is False
    assert bess_primary.evidence[0].page_number == 2
    assert wind_primary.category == "related_cross_industry_acoustic_technology"
    assert wind_primary.direct_bess_evidence is False
    assert wind_primary.cross_industry is True
    assert bess_primary.score < 0.94


def test_classification_does_not_treat_an_isolated_bess_mention_as_direct():
    page = ExtractedPage(
        page_number=1,
        text=(
            "Assessment and Rating of Wind Turbine Noise. "
            "The methods may also be relevant to BESS. "
            "Octave-band noise measurements are reported."
        ),
        text_status="extracted",
        text_character_count=126,
        image_count=0,
        acoustic_signals=("noise", "octave-band"),
        review_flags=(),
    )

    result = classify_pages([page])
    primary = next(item for item in result if item.is_primary)

    assert primary.category == "related_cross_industry_acoustic_technology"
    assert primary.direct_bess_evidence is False
    assert primary.review_status == "needs_review"
    assert primary.score < 0.94


def test_classification_marks_unextractable_relevance_for_review():
    page = ExtractedPage(
        page_number=1,
        text="",
        text_status="image_heavy",
        text_character_count=0,
        image_count=1,
        acoustic_signals=(),
        review_flags=("ocr_review", "content_not_fully_assessed"),
    )

    result = classify_pages([page], ("ocr_review",))

    assert result[0].category == "relevance_unassessed"
    assert result[0].review_status == "needs_review"
    assert result[0].evidence == ()


def test_classification_distinguishes_textually_unmatched_from_unextractable():
    readable_page = ExtractedPage(
        page_number=3,
        text="A fully text-readable report describes a landscaping plan.",
        text_status="extracted",
        text_character_count=58,
        image_count=0,
        acoustic_signals=(),
        review_flags=(),
    )

    result = classify_pages([readable_page])

    assert result[0].category == "unrelated_or_irrelevant"
    assert result[0].evidence[0].page_number == 3
    assert result[0].evidence[0].evidence_status == "not_found"


def test_additive_schema_backup_preserves_phase1_records(tmp_path):
    database_path = tmp_path / "collection.sqlite3"
    database = Database(database_path)
    document_id, _ = database.add_candidate(
        Candidate("https://example.org/preserved.pdf", "Preserved", "query", "manual")
    )
    request_id = database.record_api_request("mock", "query")
    database.finish_api_request(request_id, "succeeded", result_count=1)
    original_stats = database.statistics()

    store = DocumentStore(database)
    store.initialize()

    assert database.get_document(document_id)["title"] == "Preserved"
    assert database.statistics() == original_stats
    assert store.statistics()["processing_runs"] == 0
    with database.connect() as connection:
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert {
        "document_processing",
        "extracted_pages",
        "extracted_tables",
        "document_classifications",
        "source_evidence",
        "equipment",
        "acoustic_measurements",
        "mitigation_strategies",
        "provider_processing_eligibility_events",
    }.issubset(table_names)
    with database.connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(extracted_tables)")
        }
    assert {
        "logical_cells_json",
        "frequency_bands_json",
        "measurement_type_evidence_json",
        "alignment_status",
        "review_flags_json",
    }.issubset(columns)
    backup_path = database_path.with_name(database_path.name + "-phase2-v1-backup")
    assert backup_path.is_file()
    with sqlite3.connect(backup_path) as backup:
        assert backup.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert backup.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        assert backup.execute("SELECT COUNT(*) FROM api_requests").fetchone()[0] == 1


def test_parse_is_cached_by_document_hash_and_preserves_structured_cells(tmp_path):
    pdf_path = _make_table_pdf(tmp_path / "source.pdf")
    database = Database(tmp_path / "collection.sqlite3")
    document_id = _record_download(database, pdf_path)
    service = DocumentIntelligence(database)

    first = service.parse_documents()
    second = service.parse_documents()

    assert first.processed == 1
    assert first.failed == 0
    assert second.processed == 0
    assert second.skipped == 1
    with database.connect() as connection:
        row = connection.execute(
            "SELECT page_number, row_count, column_count, cells_json, column_headers_json, "
            "frequency_band_labels_json, units_json, logical_cells_json, "
            "frequency_bands_json, alignment_status FROM extracted_tables "
            "WHERE document_id = ?",
            (document_id,),
        ).fetchone()
    assert row["page_number"] == 1
    assert row["row_count"] == 3
    assert row["column_count"] == 2
    cells = json.loads(row["cells_json"])
    assert any(
        cell["row_index"] == 1
        and cell["column_index"] == 0
        and cell["text"] == "63"
        for cell in cells
    )
    assert json.loads(row["column_headers_json"])[0] == "Frequency band (Hz)"
    assert "63" in json.loads(row["frequency_band_labels_json"])
    assert "Hz" in json.loads(row["units_json"])
    assert "dB(A)" in json.loads(row["units_json"])
    assert json.loads(row["logical_cells_json"])
    assert json.loads(row["frequency_bands_json"])
    assert row["alignment_status"] in {"aligned", "reconstructed"}


def test_classification_persists_page_citations_and_evidence_status(tmp_path):
    pdf_path = _make_text_pdf(
        tmp_path / "assessment.pdf",
        "Battery energy storage noise impact assessment with sound pressure levels.",
    )
    database = Database(tmp_path / "collection.sqlite3")
    document_id = _record_download(
        database,
        pdf_path,
        title="Battery energy storage acoustic assessment",
    )
    service = DocumentIntelligence(database)

    service.parse_documents()
    summary = service.classify_documents()

    assert summary.processed == 1
    with database.connect() as connection:
        category = connection.execute(
            "SELECT category, is_primary FROM document_classifications "
            "WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        evidence = connection.execute(
            "SELECT page_number, evidence_status, evidence_text FROM source_evidence "
            "WHERE document_id = ? AND evidence_type = 'classification'",
            (document_id,),
        ).fetchall()
    assert category["category"] == "bess_acoustic_impact_assessment"
    assert category["is_primary"] == 1
    assert evidence
    assert all(row["page_number"] == 1 for row in evidence)
    assert all(row["evidence_status"] == "present" for row in evidence)


def test_intelligence_statistics_include_classification_review_flags(tmp_path):
    pdf_path = _make_text_pdf(
        tmp_path / "irrelevant.pdf",
        "A fully readable landscaping plan with no acoustic references.",
    )
    database = Database(tmp_path / "collection.sqlite3")
    _record_download(database, pdf_path)
    service = DocumentIntelligence(database)

    service.parse_documents()
    service.classify_documents()

    stats = service.store.statistics()
    assert stats["review_required_documents"] == 1
    assert stats["outstanding_review_items"] > 0
    repeated_parse = service.parse_documents()
    repeated_classification = service.classify_documents()
    assert repeated_parse.skipped == 1
    assert repeated_classification.skipped == 1
    assert repeated_classification.outstanding_review == stats["outstanding_review_items"]


def test_parser_handles_hash_mismatch_without_parsing(tmp_path):
    pdf_path = _make_text_pdf(tmp_path / "changed.pdf")
    database = Database(tmp_path / "collection.sqlite3")
    document_id, _ = database.add_candidate(
        Candidate("https://example.org/changed.pdf", "Changed", "query", "manual")
    )
    database.save_download(
        document_id,
        DownloadResult(
            status="downloaded",
            local_path=str(pdf_path),
            file_size=pdf_path.stat().st_size,
            sha256="0" * 64,
        ),
    )

    summary = DocumentIntelligence(database).parse_documents()

    assert summary.failed == 1
    with database.connect() as connection:
        row = connection.execute(
            "SELECT status, review_flags_json, warnings_json FROM document_processing "
            "WHERE document_id = ?",
            (document_id,),
        ).fetchone()
    assert row["status"] == "failed"
    assert "source_hash_mismatch" in row["review_flags_json"]
    assert "stored_hash_mismatch" in row["warnings_json"]


def test_failed_parse_is_classified_as_unassessed(tmp_path):
    malformed = tmp_path / "malformed.pdf"
    malformed.write_bytes(b"not a PDF")
    database = Database(tmp_path / "collection.sqlite3")
    document_id = _record_download(database, malformed)
    service = DocumentIntelligence(database)

    parsed = service.parse_documents()
    classified = service.classify_documents()

    assert parsed.failed == 1
    assert classified.processed == 1
    with database.connect() as connection:
        row = connection.execute(
            "SELECT category, review_status FROM document_classifications "
            "WHERE document_id = ?",
            (document_id,),
        ).fetchone()
    assert row["category"] == "relevance_unassessed"
    assert row["review_status"] == "needs_review"
