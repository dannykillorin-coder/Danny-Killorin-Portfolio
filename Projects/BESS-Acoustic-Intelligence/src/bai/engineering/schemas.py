from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


MetricType = Literal[
    "sound_power",
    "sound_pressure",
    "sound_reduction_index",
    "insertion_loss",
    "unknown",
]
RecordKind = Literal[
    "equipment_sound_emission",
    "ambient_environmental_measurement",
    "predicted_receptor_noise_level",
    "regulatory_noise_criterion",
    "noise_mitigation_performance",
    "building_sound_insulation_performance",
    "general_acoustic_reference_example",
    "unknown_or_ambiguous",
]
BessRelevance = Literal[
    "bess_specific",
    "bess_regulatory_context",
    "cross_industry",
    "general_reference",
    "unrelated",
    "uncertain",
]
SourceEntityRole = Literal[
    "equipment",
    "noise_source",
    "receiver",
    "monitoring_location",
    "time_period",
    "environment_description",
    "regulatory_category",
    "building_configuration",
    "unknown",
]
EvidenceRole = Literal[
    "measurement_value",
    "frequency_band",
    "metric_context",
    "equipment_label",
    "unit",
    "basis",
    "weighting",
    "distance",
    "tonal_frequency",
    "uncertainty",
    "scope",
    "operating_condition",
    "operating_load",
    "fan_speed",
    "mitigation",
    "receiver_label",
    "monitoring_location",
    "time_period",
    "environment_description",
    "regulatory_category",
    "building_configuration",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectInformation(StrictModel):
    project_name: str | None = None
    geographic_location: str | None = None
    facility_power_mw: Decimal | None = None
    storage_capacity_mwh: Decimal | None = None
    assessment_date: str | None = None
    document_type: str | None = None

    @model_validator(mode="after")
    def require_finite_capacity(self) -> ProjectInformation:
        for value in (self.facility_power_mw, self.storage_capacity_mwh):
            if value is not None and (not value.is_finite() or value <= 0):
                raise ValueError("Project capacity values must be finite and positive")
        return self


class EquipmentInformation(StrictModel):
    source_label: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    equipment_category: str | None = None
    source_type: str | None = None
    unit_count: int | None = Field(default=None, gt=0)
    operating_conditions: str | None = None


class FrequencyBandInformation(StrictModel):
    center_hz: Decimal
    source_label: str
    bandwidth: Literal["octave", "one_third_octave", "narrowband", "unknown"] = "unknown"
    source_bandwidth_text: str | None = None


class AcousticMeasurement(StrictModel):
    value: Decimal | None = None
    source_value_text: str = Field(min_length=1)
    comparator: Literal["exact", "lt", "lte", "gt", "gte", "range", "unknown"] = "unknown"
    lower_bound: Decimal | None = None
    upper_bound: Decimal | None = None
    units: str | None = None
    source_unit_text: str | None = None
    metric_descriptor: str | None = None
    metric_type: MetricType = "unknown"
    weighting: Literal["A", "C", "Z", "unweighted", "unknown"] = "unknown"
    measurement_distance: Decimal | None = None
    distance_units: str | None = None
    frequency_band: FrequencyBandInformation | None = None
    tonal_frequency_hz: Decimal | None = None
    operating_load: str | None = None
    fan_speed: str | None = None
    value_basis: Literal["measured", "predicted", "manufacturer_supplied", "unknown"] = "unknown"
    aggregation_basis: Literal["per_unit", "aggregate_facility", "unknown"] = "unknown"
    operating_conditions: str | None = None
    operating_conditions_status: Literal[
        "reported", "not_reported", "not_applicable", "ambiguous"
    ] = "not_reported"
    uncertainty: Decimal | None = None
    extraction_confidence: Decimal | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Extraction confidence only; it is not evidence of engineering correctness.",
    )
    transcription_status: Literal["candidate", "human_verified"] = "candidate"
    comparison_eligibility: Literal["not_assessed", "eligible", "ineligible"] = "not_assessed"

    @model_validator(mode="after")
    def validate_values(self) -> AcousticMeasurement:
        numeric_values = (
            self.value,
            self.lower_bound,
            self.upper_bound,
            self.measurement_distance,
            self.tonal_frequency_hz,
            self.uncertainty,
            self.frequency_band.center_hz if self.frequency_band else None,
        )
        if any(value is not None and not value.is_finite() for value in numeric_values):
            raise ValueError("Engineering numeric values must be finite")
        if self.comparator == "exact" and self.value is None:
            raise ValueError("Exact measurements require a numeric value")
        if self.comparator in {"lt", "lte"} and (
            self.upper_bound is None or self.value is not None or self.lower_bound is not None
        ):
            raise ValueError("Upper-bound measurements require only an upper bound")
        if self.comparator in {"gt", "gte"} and (
            self.lower_bound is None or self.value is not None or self.upper_bound is not None
        ):
            raise ValueError("Lower-bound measurements require only a lower bound")
        if self.comparator == "range" and (
            self.lower_bound is None
            or self.upper_bound is None
            or self.value is not None
            or self.lower_bound > self.upper_bound
        ):
            raise ValueError("Range measurements require ordered lower and upper bounds")
        if self.operating_conditions_status == "reported" and not self.operating_conditions:
            raise ValueError("Reported operating conditions require source text")
        if self.operating_conditions_status == "not_applicable" and self.operating_conditions:
            raise ValueError("Not-applicable operating conditions must not include source text")
        return self

    @model_validator(mode="after")
    def validate_comparison_eligibility(self) -> AcousticMeasurement:
        if self.comparison_eligibility != "eligible":
            return self
        required = {
            "numeric value": self.value is not None,
            "exact numeric value": self.comparator == "exact",
            "units": bool(self.units),
            "measurement type": self.metric_type != "unknown",
            "weighting": self.weighting != "unknown",
            "measurement basis": self.value_basis != "unknown",
            "aggregation scope": self.aggregation_basis != "unknown",
            "operating conditions": self.operating_conditions_status
            in {"reported", "not_applicable"},
        }
        if self.frequency_band is not None:
            required["frequency bandwidth"] = self.frequency_band.bandwidth != "unknown"
        missing = [label for label, satisfied in required.items() if not satisfied]
        if missing:
            raise ValueError(
                "Comparison-eligible measurements require explicit " + ", ".join(missing)
            )
        return self


class NoiseMitigation(StrictModel):
    mitigation_type: str | None = None
    equipment_targeted: str | None = None
    reported_attenuation: Decimal | None = None
    attenuation_units: str | None = None
    frequency_specific_performance: str | None = None
    cooling_or_airflow_penalty: str | None = None
    installation_footprint: str | None = None
    reported_cost: Decimal | None = None
    cost_units: str | None = None
    deployment_evidence: str | None = None

    @model_validator(mode="after")
    def require_finite_cost_and_attenuation(self) -> NoiseMitigation:
        if self.reported_attenuation is not None and not self.reported_attenuation.is_finite():
            raise ValueError("Reported attenuation must be finite")
        if self.reported_cost is not None and (
            not self.reported_cost.is_finite() or self.reported_cost < 0
        ):
            raise ValueError("Reported cost must be finite and non-negative")
        return self


class DocumentSource(StrictModel):
    document_id: int = Field(gt=0)
    title: str
    filename: str
    original_url: str
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")


class SourceEvidenceReference(StrictModel):
    evidence_id: int = Field(gt=0)
    role: EvidenceRole
    document_id: int = Field(gt=0)
    page_number: int = Field(gt=0)
    table_id: int | None = Field(default=None, gt=0)
    row_index: int | None = Field(default=None, ge=0)
    column_index: int | None = Field(default=None, ge=0)
    quote: str = Field(min_length=1)


class PageContextEvidence(StrictModel):
    document_id: int = Field(gt=0)
    page_number: int = Field(gt=0)
    role: Literal["caption", "nearby_context", "footnote"]
    quote: str = Field(min_length=1)


class HumanReviewedEvidence(StrictModel):
    document_id: int = Field(gt=0)
    page_number: int = Field(gt=0)
    table_reference: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    verification_method: Literal["text_match", "visual_pdf_review"]


class BenchmarkCorrection(StrictModel):
    measurement: AcousticMeasurement
    human_review_evidence: tuple[HumanReviewedEvidence, ...] = Field(min_length=1)


class BenchmarkCandidate(StrictModel):
    candidate_id: str = Field(pattern=r"^cand-[a-f0-9]{16}$")
    generator_version: Literal["stage-b1.0", "stage-b1.1"] = "stage-b1.0"
    source: DocumentSource
    page_number: int = Field(gt=0)
    table_id: int = Field(gt=0)
    source_table_reference: str | None = None
    record_kind: RecordKind = "unknown_or_ambiguous"
    bess_relevance: BessRelevance = "uncertain"
    source_entity_role: SourceEntityRole = "unknown"
    source_entity_label: str | None = None
    equipment: EquipmentInformation = EquipmentInformation()
    measurement: AcousticMeasurement
    evidence: tuple[SourceEvidenceReference, ...] = Field(min_length=1)
    page_context: tuple[PageContextEvidence, ...] = ()
    review_flags: tuple[str, ...] = ()
    candidate_status: Literal["pending_review"] = "pending_review"

    @model_validator(mode="after")
    def require_numeric_evidence(self) -> BenchmarkCandidate:
        has_numeric_claim = any(
            value is not None
            for value in (
                self.measurement.value,
                self.measurement.lower_bound,
                self.measurement.upper_bound,
            )
        )
        if has_numeric_claim and not any(
            item.role == "measurement_value" for item in self.evidence
        ):
            raise ValueError("A numerical measurement requires page-level value evidence")
        if self.generator_version == "stage-b1.1":
            measurement = self.measurement
            if measurement.value is not None and measurement.comparator != "exact":
                raise ValueError("B1.1 exact numeric values must be explicitly classified")
            if measurement.value is None and measurement.comparator == "exact":
                raise ValueError("B1.1 exact numeric values cannot be missing")
        for item in self.evidence:
            if item.document_id != self.source.document_id:
                raise ValueError("Evidence and source document IDs must match")
            if item.page_number != self.page_number:
                raise ValueError("All candidate evidence must cite the candidate's source page")
            if item.table_id != self.table_id:
                raise ValueError("All candidate table evidence must cite the source table")
        for item in self.page_context:
            if item.document_id != self.source.document_id:
                raise ValueError("Page context and source document IDs must match")
            if item.page_number != self.page_number:
                raise ValueError("Page context must cite the candidate's source page")
        return self


class BenchmarkReviewItem(StrictModel):
    review_item_id: str = Field(pattern=r"^review-[a-f0-9]{16}$")
    generator_version: Literal["stage-b1.0", "stage-b1.1"]
    source: DocumentSource
    page_number: int = Field(gt=0)
    table_id: int = Field(gt=0)
    table_index: int = Field(ge=0)
    row_index: int | None = Field(default=None, ge=0)
    column_index: int | None = Field(default=None, ge=0)
    source_table_reference: str | None = None
    reason: str = Field(min_length=1)
    excerpt: str = ""
    status: Literal["open", "resolved"] = "open"


class GoldStandardRecord(StrictModel):
    candidate_id: str
    generator_version: Literal["stage-b1.0", "stage-b1.1"] = "stage-b1.0"
    source: DocumentSource
    page_number: int = Field(gt=0)
    table_id: int = Field(gt=0)
    source_table_reference: str | None = None
    record_kind: RecordKind = "unknown_or_ambiguous"
    bess_relevance: BessRelevance = "uncertain"
    source_entity_role: SourceEntityRole = "unknown"
    source_entity_label: str | None = None
    equipment: EquipmentInformation = EquipmentInformation()
    measurement: AcousticMeasurement
    evidence: tuple[SourceEvidenceReference, ...] = Field(min_length=1)
    page_context: tuple[PageContextEvidence, ...] = ()
    human_review_evidence: tuple[HumanReviewedEvidence, ...] = ()
    approved_by: str = Field(min_length=1)
    approved_at: datetime


class ReviewEvent(StrictModel):
    candidate_id: str
    action: Literal["approved", "rejected", "corrected"]
    reviewer: str = Field(min_length=1)
    reason: str = ""
    recorded_at: datetime
    corrected_measurement: AcousticMeasurement | None = None
    human_review_evidence: tuple[HumanReviewedEvidence, ...] = ()

    @model_validator(mode="after")
    def corrections_require_measurement(self) -> ReviewEvent:
        if self.action == "corrected":
            if self.corrected_measurement is None or not self.human_review_evidence:
                raise ValueError(
                    "Corrections require a measurement and human-verified source evidence"
                )
        elif self.corrected_measurement is not None or self.human_review_evidence:
            raise ValueError("Only correction events may include correction evidence")
        return self
