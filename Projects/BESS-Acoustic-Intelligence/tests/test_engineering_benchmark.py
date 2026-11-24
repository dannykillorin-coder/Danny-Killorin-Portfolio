import hashlib
import sqlite3
from decimal import Decimal
from pathlib import Path

import pymupdf
import pytest
from pydantic import ValidationError

from bai.database import Database
from bai.documents.parser import PDFParser
from bai.documents.store import DocumentStore
from bai.engineering.benchmark import BenchmarkGenerator
from bai.engineering.benchmark_store import BenchmarkStore
from bai.engineering.semantic_benchmark import _bess_relevance, parse_source_value
from bai.engineering.schemas import (
    AcousticMeasurement,
    BenchmarkReviewItem,
    DocumentSource,
    HumanReviewedEvidence,
)
from bai.engineering.validation import (
    EvidenceValidationError,
    parse_unambiguous_number,
    validate_candidate_evidence,
)
from bai.models import Candidate, DownloadResult
from bai.models import Settings


def _make_vertical_spectrum_pdf(path):
    document = pymupdf.open()
    page = document.new_page(width=400, height=320)
    page.insert_text((40, 55), "Table 1: octave-band sound pressure levels")
    x_positions = (40, 200, 360)
    y_positions = (90, 130, 170, 210)
    for x in x_positions:
        page.draw_line((x, y_positions[0]), (x, y_positions[-1]))
    for y in y_positions:
        page.draw_line((x_positions[0], y), (x_positions[-1], y))
    rows = (
        ("Frequency band (Hz)", "Sound pressure level dB(A)"),
        ("31.5", "65"),
        ("63", "70"),
    )
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            page.insert_text(
                (x_positions[column_index] + 5, y_positions[row_index] + 25),
                value,
                fontsize=9,
            )
    document.set_metadata({"title": "Verified spectrum fixture"})
    document.save(path)
    document.close()
    return path


def _parsed_database(tmp_path):
    pdf_path = _make_vertical_spectrum_pdf(tmp_path / "spectrum.pdf")
    digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    database = Database(tmp_path / "collection.sqlite3")
    document_id, _ = database.add_candidate(
        Candidate(
            url="https://example.org/spectrum.pdf",
            title="Verified spectrum fixture",
            query="manual",
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
    store = DocumentStore(database)
    store.initialize()
    result = PDFParser().parse(pdf_path)
    assert result.tables
    store.save_parse(document_id, result)
    return database, document_id


def test_comparison_eligibility_requires_explicit_engineering_context():
    with pytest.raises(ValidationError, match="Comparison-eligible"):
        AcousticMeasurement(
            value=Decimal("65"),
            source_value_text="65",
            comparison_eligibility="eligible",
        )

    measurement = AcousticMeasurement(
        source_value_text="not stated",
        value=None,
        comparison_eligibility="not_assessed",
    )
    assert measurement.value is None


def test_numeric_parser_rejects_ranges_and_ambiguous_grouping():
    assert parse_unambiguous_number("31.5 dB(A)") == Decimal("31.5")
    assert parse_unambiguous_number("50-55 dB") is None
    assert parse_unambiguous_number("1,000 dB") is None


def test_semantic_numeric_parser_preserves_bounds_and_ranges():
    bounded = parse_source_value("≤10")
    assert bounded is not None
    assert bounded.comparator == "lte"
    assert bounded.value is None
    assert bounded.upper_bound == Decimal("10")

    ranged = parse_source_value("10-15 dB")
    assert ranged is not None
    assert ranged.comparator == "range"
    assert ranged.lower_bound == Decimal("10")
    assert ranged.upper_bound == Decimal("15")


def test_isolated_bess_mentions_do_not_establish_direct_relevance():
    assert (
        _bess_relevance(
            "equipment_sound_emission",
            "BESS is mentioned in an unrelated section.",
            "Industrial fan",
            "Sound power level",
            "General acoustic report",
            "https://example.org/report.pdf",
        )
        == "uncertain"
    )
    assert (
        _bess_relevance(
            "equipment_sound_emission",
            "Battery storage units are listed in Table 2.",
            "Inverter",
            "Sound power level",
            "General acoustic report",
            "https://example.org/report.pdf",
        )
        == "bess_specific"
    )


def test_review_items_are_persistent_and_idempotent(tmp_path):
    store = BenchmarkStore(tmp_path / "benchmarks.sqlite3")
    store.initialize()
    item = BenchmarkReviewItem(
        review_item_id="review-0123456789abcdef",
        generator_version="stage-b1.1",
        source=DocumentSource(
            document_id=1,
            title="Regression fixture",
            filename="fixture.pdf",
            original_url="https://example.org/fixture.pdf",
            sha256="a" * 64,
        ),
        page_number=1,
        table_id=1,
        table_index=0,
        source_table_reference="Table 1",
        reason="ambiguous_table_alignment",
    )

    assert store.add_review_items([item]) == (1, 0)
    assert store.add_review_items([item]) == (0, 1)
    assert store.review_items("open") == [item]


def test_candidate_generation_maps_vertical_frequency_rows_to_values(tmp_path):
    database, _ = _parsed_database(tmp_path)
    benchmark_store = BenchmarkStore(tmp_path / "benchmarks.sqlite3")

    added, existing, selected = BenchmarkGenerator(
        database,
        benchmark_store,
    ).generate()

    assert (added, existing, selected) == (2, 0, 2)
    candidates = [
        candidate
        for _, _, candidate in benchmark_store.list_candidates()
    ]
    assert {candidate.measurement.value for candidate in candidates} == {
        Decimal("65"),
        Decimal("70"),
    }
    assert {
        candidate.measurement.frequency_band.center_hz
        for candidate in candidates
    } == {Decimal("31.5"), Decimal("63.0")}
    assert all(candidate.measurement.metric_type == "sound_pressure" for candidate in candidates)
    assert all(candidate.candidate_status == "pending_review" for candidate in candidates)
    assert BenchmarkGenerator(database, benchmark_store).generate() == (0, 2, 2)
    original = candidates[0]
    reparsed = original.model_copy(
        update={
            "candidate_id": f"cand-{int(original.candidate_id[5:], 16) + 1:016x}",
            "table_id": original.table_id + 100,
            "evidence": tuple(
                item.model_copy(
                    update={
                        "evidence_id": item.evidence_id + 100,
                        "table_id": item.table_id + 100,
                    }
                )
                for item in original.evidence
            ),
        }
    )
    assert benchmark_store.add_candidates([reparsed]) == (0, 1)


def test_unsupported_value_fails_and_human_correction_is_append_only(tmp_path):
    database, _ = _parsed_database(tmp_path)
    benchmark_store = BenchmarkStore(tmp_path / "benchmarks.sqlite3")
    BenchmarkGenerator(database, benchmark_store).generate()
    candidate_id, _, candidate = benchmark_store.list_candidates()[0]
    corrected = AcousticMeasurement.model_validate(
        {
            **candidate.measurement.model_dump(),
            "value": Decimal("66"),
            "source_value_text": "66 dB(A)",
        }
    )
    with pytest.raises(EvidenceValidationError, match="not explicitly supported"):
        validate_candidate_evidence(candidate, database, corrected)

    human_evidence = (
        HumanReviewedEvidence(
            document_id=candidate.source.document_id,
            page_number=candidate.page_number,
            table_reference="Table 1",
            quote="66 dB(A)",
            verification_method="visual_pdf_review",
        ),
    )
    validate_candidate_evidence(
        candidate,
        database,
        corrected,
        human_evidence,
    )
    benchmark_store.record_correction(
        candidate_id,
        corrected,
        "reviewer",
        "Verified against the PDF page",
        human_evidence,
    )
    current_measurement, current_evidence = benchmark_store.current_review(candidate_id)
    assert current_measurement.value == Decimal("66")
    assert current_evidence == human_evidence
    assert benchmark_store.get_candidate(candidate_id).measurement.value == candidate.measurement.value

    benchmark_store.approve(
        candidate_id,
        "reviewer",
        "Approved after source review",
        current_measurement,
        current_evidence,
    )
    assert benchmark_store.statistics()["gold_records"] == 1
    assert [event.action for event in benchmark_store.review_history(candidate_id)] == [
        "corrected",
        "approved",
    ]
    with sqlite3.connect(benchmark_store.path) as connection:
        original_payload = connection.execute(
            "SELECT payload_json FROM benchmark_candidates WHERE candidate_id = ?",
            (candidate_id,),
        ).fetchone()[0]
    assert '"source_value_text":"66 dB(A)"' not in original_payload


def test_provider_processing_is_default_deny_and_auditable(tmp_path):
    database = Database(tmp_path / "collection.sqlite3")
    document_id, _ = database.add_candidate(
        Candidate(
            url="https://example.org/private.pdf",
            title="Restricted research",
            query="manual",
            provider="manual",
        )
    )
    store = DocumentStore(database)
    store.initialize()
    assert store.provider_eligibility()[0]["eligible"] == 0
    assert not store.is_provider_processing_eligible(document_id)

    store.set_provider_eligibility(
        document_id,
        True,
        "reviewer",
        "Explicit authorization for this document",
    )
    assert store.is_provider_processing_eligible(document_id)
    store.set_provider_eligibility(
        document_id,
        False,
        "reviewer",
        "Authorization withdrawn",
    )
    assert store.provider_eligibility()[0]["eligible"] == 0
    assert not store.is_provider_processing_eligible(document_id)
    with database.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM provider_processing_eligibility_events"
        ).fetchone()[0] == 2


def test_cli_generates_displays_and_approves_without_provider_calls(
    tmp_path,
    monkeypatch,
    capsys,
):
    database, _ = _parsed_database(tmp_path)
    benchmark_path = tmp_path / "benchmarks.sqlite3"

    def resolve_local_path(value):
        path = Path(value)
        return path if path.is_absolute() else tmp_path / path

    monkeypatch.setattr("bai.cli.load_settings", lambda: Settings(database_path=str(database.path)))
    monkeypatch.setattr("bai.cli.resolve_project_path", resolve_local_path)
    monkeypatch.setattr("bai.cli.DEFAULT_BENCHMARK_PATH", benchmark_path)

    from bai.cli import main

    assert main(["benchmark", "generate", "--limit", "1"]) == 0
    store = BenchmarkStore(benchmark_path)
    candidate_id = store.list_candidates()[0][0]
    assert main(["benchmark", "show", candidate_id]) == 0
    assert candidate_id in capsys.readouterr().out
    assert (
        main(
            [
                "benchmark",
                "review",
                candidate_id,
                "--action",
                "approve",
                "--reviewer",
                "reviewer",
                "--reason",
                "Verified against source",
            ]
        )
        == 0
    )
    assert store.statistics()["gold_records"] == 1
