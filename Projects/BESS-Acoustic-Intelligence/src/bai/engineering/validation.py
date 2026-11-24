from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation

from bai.database import Database
from bai.engineering.schemas import (
    AcousticMeasurement,
    BenchmarkCandidate,
    HumanReviewedEvidence,
)


_NUMBER = re.compile(r"(?<![\w.,])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?![\w.,])")
_RANGE = re.compile(r"\d\s*[-\u2012-\u2015]\s*\d|±")
_BOUND_PREFIX = re.compile(r"^\s*(?:<=|>=|<|>|≤|≥)")
_SOURCE_BOUND = re.compile(
    r"^\s*(<=|>=|<|>|≤|≥)\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\b"
)
_SOURCE_RANGE = re.compile(
    r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:-|–|—|to)\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\b",
    re.I,
)
_METRIC_TERMS = {
    "sound_power": re.compile(r"\bsound\s+power\b|\bpower\s+level\b", re.I),
    "sound_pressure": re.compile(r"\bsound\s+pressure\b|\bpressure\s+level\b|\bSPL\b", re.I),
    "sound_reduction_index": re.compile(r"\bsound\s+reduction\s+index\b|\bSRI\b", re.I),
    "insertion_loss": re.compile(r"\binsertion\s+loss\b", re.I),
}


class EvidenceValidationError(ValueError):
    """Raised when an engineering value cannot be tied to persisted source evidence."""


def parse_unambiguous_number(text: str) -> Decimal | None:
    if _RANGE.search(text) or _BOUND_PREFIX.search(text):
        return None
    matches = _NUMBER.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0])
    except InvalidOperation:
        return None


def validate_candidate_evidence(
    candidate: BenchmarkCandidate,
    database: Database,
    measurement: AcousticMeasurement | None = None,
    human_evidence: tuple[HumanReviewedEvidence, ...] = (),
) -> None:
    record = measurement or candidate.measurement
    with database.connect() as connection:
        document = connection.execute(
            """
            SELECT title, source_url, local_path, sha256
            FROM documents WHERE id = ?
            """,
            (candidate.source.document_id,),
        ).fetchone()
        if document is None:
            raise EvidenceValidationError("Source document no longer exists")
        if document["sha256"] != candidate.source.sha256:
            raise EvidenceValidationError("Source document hash does not match the candidate")
        if (document["title"] or "") != candidate.source.title:
            raise EvidenceValidationError("Source document title does not match the candidate")
        if document["source_url"] != candidate.source.original_url:
            raise EvidenceValidationError("Source URL does not match the candidate")
        if not document["local_path"] or document["local_path"].split("\\")[-1].split("/")[-1] != candidate.source.filename:
            raise EvidenceValidationError("Source filename does not match the candidate")

        for citation in candidate.evidence:
            row = connection.execute(
                """
                SELECT document_id, page_number, table_id, row_index, column_index,
                       evidence_text, evidence_status
                FROM source_evidence WHERE id = ?
                """,
                (citation.evidence_id,),
            ).fetchone()
            if row is None:
                raise EvidenceValidationError(
                    f"Evidence ID {citation.evidence_id} does not exist"
                )
            if (
                row["document_id"] != citation.document_id
                or row["page_number"] != citation.page_number
                or row["table_id"] != citation.table_id
                or row["row_index"] != citation.row_index
                or row["column_index"] != citation.column_index
                or row["evidence_text"] != citation.quote
                or row["evidence_status"] != "present"
            ):
                raise EvidenceValidationError(
                    f"Evidence ID {citation.evidence_id} does not match its stored source"
                )
        context_quotes = []
        for context in candidate.page_context:
            page = connection.execute(
                """
                SELECT text FROM extracted_pages
                WHERE document_id = ? AND page_number = ?
                ORDER BY processing_id DESC LIMIT 1
                """,
                (context.document_id, context.page_number),
            ).fetchone()
            if page is None or context.quote not in page["text"]:
                raise EvidenceValidationError(
                    "Page context quote is not present on its cited parsed source page"
                )
            context_quotes.append(context.quote)
        human_quotes = []
        for item in human_evidence:
            if item.document_id != candidate.source.document_id:
                raise EvidenceValidationError(
                    "Human-reviewed evidence must belong to the candidate's source document"
                )
            page = connection.execute(
                """
                SELECT text FROM extracted_pages
                WHERE document_id = ? AND page_number = ?
                ORDER BY processing_id DESC LIMIT 1
                """,
                (item.document_id, item.page_number),
            ).fetchone()
            if page is None:
                raise EvidenceValidationError(
                    f"Human-reviewed source page {item.page_number} is not in the parsed document"
                )
            if (
                item.verification_method == "text_match"
                and item.quote not in page["text"]
            ):
                raise EvidenceValidationError(
                    "Text-match evidence quote is not present on its cited parsed page"
                )
            human_quotes.append(item.quote)
        if human_evidence and not any(
            record.source_value_text in quote for quote in human_quotes
        ):
            raise EvidenceValidationError(
                "Corrected source value text must be preserved in the human-reviewed quote"
            )

        all_quotes = (
            [item.quote for item in candidate.evidence]
            + context_quotes
            + human_quotes
        )
        if candidate.equipment.source_label and not any(
            item.role == "equipment_label"
            and candidate.equipment.source_label == item.quote
            for item in candidate.evidence
        ) and not any(candidate.equipment.source_label in quote for quote in human_quotes):
            raise EvidenceValidationError("Equipment label is not supported by cited evidence")
        for field_name in ("manufacturer", "model", "equipment_category", "source_type"):
            field_value = getattr(candidate.equipment, field_name)
            if field_value and not any(field_value in quote for quote in all_quotes):
                raise EvidenceValidationError(
                    f"Equipment {field_name.replace('_', ' ')} is not supported by cited evidence"
                )
        if candidate.equipment.unit_count is not None and not any(
            parse_unambiguous_number(quote) == Decimal(candidate.equipment.unit_count)
            for quote in all_quotes
        ):
            raise EvidenceValidationError("Equipment unit count is not supported by cited evidence")
        if candidate.source_entity_label:
            if not any(
                item.role == _source_entity_evidence_role(candidate.source_entity_role)
                for item in candidate.evidence
            ):
                raise EvidenceValidationError("Source entity label lacks role-specific evidence")
            label_tokens = re.findall(r"[\w]+", candidate.source_entity_label.casefold())
            source_tokens = re.findall(
                r"[\w]+", " ".join(all_quotes).casefold()
            )
            if not label_tokens or not all(token in source_tokens for token in label_tokens):
                raise EvidenceValidationError(
                    "Source entity label is not supported by cited table/page evidence"
                )

        if not any(item.role == "measurement_value" for item in candidate.evidence):
            raise EvidenceValidationError("Numerical finding has no cited value cell")
        value_citations = [
            item for item in candidate.evidence if item.role == "measurement_value"
        ]
        if record.value is not None:
            supported_by_parsed_cell = any(
                parse_unambiguous_number(item.quote) == record.value
                for item in value_citations
            )
            supported_by_human_quote = any(
                parse_unambiguous_number(quote) == record.value
                and record.source_value_text in quote
                for quote in human_quotes
            )
            if not supported_by_parsed_cell and not supported_by_human_quote:
                raise EvidenceValidationError(
                    "Numerical value is not explicitly supported by its cited source cell"
                )
            if parse_unambiguous_number(record.source_value_text) != record.value:
                raise EvidenceValidationError(
                    "The preserved source value text does not support the numerical value"
                )
        _validate_numeric_interpretation(record)
        if record.units:
            unit_source = record.source_unit_text or record.units
            if not any(unit_source in quote for quote in all_quotes):
                raise EvidenceValidationError("Reported units are not present in cited evidence")
            if record.units not in unit_source:
                raise EvidenceValidationError("Reported units do not match their source wording")
        if record.metric_descriptor and not any(
            record.metric_descriptor in quote for quote in all_quotes
        ):
            raise EvidenceValidationError("Metric descriptor is not preserved in source evidence")

        if record.metric_type != "unknown":
            matcher = _METRIC_TERMS[record.metric_type]
            if not any(
                item.role == "metric_context" and matcher.search(item.quote)
                for item in candidate.evidence
            ) and not any(matcher.search(quote) for quote in all_quotes):
                raise EvidenceValidationError(
                    f"Measurement type {record.metric_type} is not explicit in cited evidence"
                )
        if record.weighting != "unknown":
            expected = {
                "A": re.compile(r"dB\s*\(\s*A\s*\)|dBA", re.I),
                "C": re.compile(r"dB\s*\(\s*C\s*\)|dBC", re.I),
                "Z": re.compile(r"dB\s*\(\s*Z\s*\)|dBZ", re.I),
                "unweighted": re.compile(r"\bunweighted\b", re.I),
            }[record.weighting]
            if not any(expected.search(item.quote) for item in candidate.evidence) and not any(
                expected.search(quote) for quote in all_quotes
            ):
                raise EvidenceValidationError("Weighting is not explicit in cited evidence")
        if record.value_basis != "unknown":
            basis_terms = {
                "measured": re.compile(
                    r"\bmeasured\b|\bmeasurements?\b|\bmonitoring\s+results?\b",
                    re.I,
                ),
                "predicted": re.compile(r"\bpredict(?:ed|ion)\b|\bmodelled\b|\bmodeled\b", re.I),
                "manufacturer_supplied": re.compile(
                    r"\bmanufacturer[- ]supplied\b|\bmanufacturer's data\b|"
                    r"\bprovided by (?:the )?manufacturer\b|"
                    r"\bsupplier noise assessment\b",
                    re.I,
                ),
            }[record.value_basis]
            if not any(
                item.role == "basis" and basis_terms.search(item.quote)
                for item in candidate.evidence
            ) and not any(basis_terms.search(quote) for quote in all_quotes):
                raise EvidenceValidationError("Value basis is not explicit in cited evidence")

        if record.measurement_distance is not None:
            if not record.distance_units or not any(
                item.role == "distance"
                and parse_unambiguous_number(item.quote) == record.measurement_distance
                and record.distance_units in item.quote
                for item in candidate.evidence
            ) and not any(
                parse_unambiguous_number(quote) == record.measurement_distance
                and record.distance_units
                and record.distance_units in quote
                for quote in human_quotes
            ):
                raise EvidenceValidationError(
                    "Measurement distance and units are not explicitly supported"
                )
        if record.tonal_frequency_hz is not None and not any(
            item.role == "tonal_frequency"
            and parse_unambiguous_number(item.quote) == record.tonal_frequency_hz
            for item in candidate.evidence
        ) and not any(
            parse_unambiguous_number(quote) == record.tonal_frequency_hz
            for quote in human_quotes
        ):
            raise EvidenceValidationError("Tonal frequency is not explicitly supported")
        if record.uncertainty is not None and not any(
            item.role == "uncertainty"
            and parse_unambiguous_number(item.quote) == record.uncertainty
            for item in candidate.evidence
        ) and not any(
            parse_unambiguous_number(quote) == record.uncertainty
            for quote in human_quotes
        ):
            raise EvidenceValidationError("Measurement uncertainty is not explicitly supported")
        if record.aggregation_basis != "unknown":
            scope_terms = {
                "per_unit": re.compile(r"\bper[\s-]+unit\b|\beach\s+unit\b", re.I),
                "aggregate_facility": re.compile(
                    r"\baggregate\b|\bfacility[- ]wide\b|\btotal\s+(?:facility|site)\b",
                    re.I,
                ),
            }[record.aggregation_basis]
            if not any(
                item.role == "scope" and scope_terms.search(item.quote)
                for item in candidate.evidence
            ) and not any(scope_terms.search(quote) for quote in human_quotes):
                raise EvidenceValidationError("Aggregation scope is not explicit in evidence")
        if record.operating_conditions_status == "reported":
            if not any(
                item.role == "operating_condition"
                and record.operating_conditions in item.quote
                for item in candidate.evidence
            ) and not any(
                record.operating_conditions in quote for quote in human_quotes
            ):
                raise EvidenceValidationError("Operating conditions are not supported by evidence")
        for field_name, role in (
            ("operating_load", "operating_load"),
            ("fan_speed", "fan_speed"),
        ):
            field_value = getattr(record, field_name)
            if field_value and not any(
                item.role == role and field_value in item.quote
                for item in candidate.evidence
            ) and not any(field_value in quote for quote in human_quotes):
                raise EvidenceValidationError(
                    f"{field_name.replace('_', ' ')} is not supported by cited evidence"
                )

        _validate_record_semantics(candidate, all_quotes)

        if record.frequency_band is not None:
            band_citations = [
                item for item in candidate.evidence if item.role == "frequency_band"
            ]
            if not any(
                item.quote == record.frequency_band.source_label
                or record.frequency_band.source_label in item.quote
                for item in band_citations
            ):
                raise EvidenceValidationError("Frequency label is not present in cited evidence")
            table = connection.execute(
                """
                SELECT frequency_bands_json FROM extracted_tables
                WHERE id = ? AND document_id = ?
                """,
                (candidate.table_id, candidate.source.document_id),
            ).fetchone()
            mappings = json.loads(table["frequency_bands_json"]) if table else []
            if not any(
                band["source_label"] == record.frequency_band.source_label
                and Decimal(str(band["center_hz"])) == record.frequency_band.center_hz
                for band in mappings
            ):
                _validate_explicit_frequency_mapping(
                    record.frequency_band.source_label,
                    record.frequency_band.center_hz,
                    band_citations,
                )
            if record.frequency_band.bandwidth != "unknown":
                bandwidth_terms = {
                    "octave": re.compile(r"(?<!third[- ])\boctave\b", re.I),
                    "one_third_octave": re.compile(r"one[- ]third octave", re.I),
                    "narrowband": re.compile(r"narrowband", re.I),
                }[record.frequency_band.bandwidth]
                if not any(bandwidth_terms.search(quote) for quote in all_quotes):
                    raise EvidenceValidationError(
                        "Frequency bandwidth is not explicit in cited evidence"
                    )


def _validate_numeric_interpretation(record: AcousticMeasurement) -> None:
    source = record.source_value_text.strip().replace("−", "-")
    bound = _SOURCE_BOUND.match(source)
    if record.comparator in {"lt", "lte", "gt", "gte"}:
        if bound is None:
            raise EvidenceValidationError(
                "A bounded value must preserve its source comparator"
            )
        operator, number = bound.groups()
        normalized_operator = operator.replace("≤", "<=").replace("≥", ">=")
        expected_operator = {
            "lt": "<",
            "lte": "<=",
            "gt": ">",
            "gte": ">=",
        }[record.comparator]
        if normalized_operator != expected_operator:
            raise EvidenceValidationError(
                "Stored comparator does not match the source value wording"
            )
        parsed = Decimal(number)
        stored = (
            record.upper_bound
            if record.comparator in {"lt", "lte"}
            else record.lower_bound
        )
        if stored != parsed:
            raise EvidenceValidationError(
                "Stored numeric bound does not match the source value"
            )
        return

    range_match = _SOURCE_RANGE.match(source)
    if record.comparator == "range":
        if range_match is None:
            raise EvidenceValidationError(
                "A range value must preserve both source endpoints"
            )
        lower, upper = (Decimal(value) for value in range_match.groups())
        if (record.lower_bound, record.upper_bound) != (lower, upper):
            raise EvidenceValidationError(
                "Stored range endpoints do not match the source value"
            )
        return

    if record.comparator == "exact":
        value = parse_unambiguous_number(source)
        if value is None or value != record.value:
            raise EvidenceValidationError(
                "Exact numeric interpretation does not match the source value"
            )


def _source_entity_evidence_role(entity_role: str) -> str:
    return {
        "equipment": "equipment_label",
        "noise_source": "equipment_label",
        "receiver": "receiver_label",
        "monitoring_location": "monitoring_location",
        "time_period": "time_period",
        "environment_description": "environment_description",
        "regulatory_category": "regulatory_category",
        "building_configuration": "building_configuration",
    }.get(entity_role, "metric_context")


def _validate_record_semantics(
    candidate: BenchmarkCandidate,
    quotes: list[str],
) -> None:
    context = " ".join(
        (candidate.source.title, candidate.source.original_url, *quotes)
    )
    if candidate.record_kind == "equipment_sound_emission":
        if not (
            re.search(r"\bsound\s+power\b|\bSWL\b", context, re.I)
            and candidate.source_entity_role == "equipment"
        ):
            raise EvidenceValidationError(
                "Equipment-emission classification requires explicit equipment sound-power evidence"
            )
    elif candidate.record_kind == "predicted_receptor_noise_level":
        if not (
            re.search(r"\bpredict(?:ed|ion)\b|\bmodelled\b|\bmodeled\b", context, re.I)
            and re.search(
                r"\b(?:receiver|receptor|NSR|property\s+boundary|property\s+line|"
                r"nearest\s+residence)\b",
                context,
                re.I,
            )
        ):
            raise EvidenceValidationError(
                "Predicted receptor classification lacks explicit prediction/receiver evidence"
            )
    elif candidate.record_kind == "regulatory_noise_criterion":
        if not re.search(
            r"\b(?:criterion|criteria|limits?|threshold|objective|permitted|"
            r"requirements?)\b",
            context,
            re.I,
        ):
            raise EvidenceValidationError(
                "Regulatory classification lacks explicit criteria or limit wording"
            )
    elif candidate.record_kind == "building_sound_insulation_performance":
        if not re.search(
            r"\b(?:Rw\s*\+\s*Ctr|façade|facade|sound\s+insulation|"
            r"building\s+sound|SRI)\b",
            context,
            re.I,
        ):
            raise EvidenceValidationError(
                "Building-insulation classification lacks explicit source wording"
            )
    elif candidate.record_kind == "general_acoustic_reference_example":
        if not re.search(
            r"\b(?:jet\s+aircraft|jet\s+flyover|wilderness|typical\s+sound|"
            r"typical\s+noise\s+sources?|acoustic\s+environments|"
            r"common\s+sound\s+sources?|reference\s+(?:sound|noise)\s+levels?)\b",
            context,
            re.I,
        ):
            raise EvidenceValidationError(
                "General-reference classification lacks an explicit reference example"
            )

    if candidate.bess_relevance == "bess_specific":
        if not re.search(
            r"\bBESS\b|battery\s+(?:energy\s+)?storage"
            r"(?:\s+(?:system|equipment|units?|containers?|enclosures?))?|"
            r"\benergy\s+storage\s+system\b",
            context,
            re.I,
        ):
            raise EvidenceValidationError(
                "BESS-specific relevance lacks explicit BESS source context"
            )
        if candidate.record_kind == "general_acoustic_reference_example":
            raise EvidenceValidationError(
                "Generic acoustic reference examples cannot be BESS-specific findings"
            )
    elif candidate.bess_relevance == "bess_regulatory_context":
        if candidate.record_kind != "regulatory_noise_criterion" or not re.search(
            r"\bBESS\b|battery\s+(?:energy\s+)?storage"
            r"(?:\s+(?:system|equipment|units?|containers?|enclosures?))?|"
            r"\benergy\s+storage\s+system\b",
            context,
            re.I,
        ):
            raise EvidenceValidationError(
                "BESS regulatory context requires both a criterion and explicit BESS context"
            )


def _validate_explicit_frequency_mapping(
    source_label: str,
    center_hz: Decimal,
    citations: list,
) -> None:
    source_pattern = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*(kHz|Hz|kilohertz|hertz|k)?\s*",
        source_label,
        re.I,
    )
    context = " ".join(item.quote for item in citations)
    if source_pattern is None or not re.search(
        r"\bfrequency\b|\bfreq\.?\b|octave[- ]band",
        context,
        re.I,
    ):
        raise EvidenceValidationError(
            "Frequency center lacks an explicit header-scoped frequency label"
        )
    number, unit = source_pattern.groups()
    if unit is None:
        unit_match = re.search(
            r"\b(kHz|Hz|kilohertz|hertz)\b",
            context,
            re.I,
        )
        if unit_match is None:
            raise EvidenceValidationError(
                "Frequency center has no explicit Hz or kHz unit"
            )
        unit = unit_match.group(1)
    expected = Decimal(number)
    if unit.casefold() in {"khz", "kilohertz", "k"}:
        expected *= Decimal("1000")
    if center_hz != expected:
        raise EvidenceValidationError(
            "Frequency center does not match the cited frequency label and unit"
        )
