from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import replace

from bai.documents.models import (
    ClassificationEvidence,
    DocumentClassification,
    ExtractedPage,
)


CLASSIFIER_VERSION = "evidence-2"

_BESS_PATTERN = re.compile(
    r"\b(?:BESS|battery\s+energy\s+storage|battery\s+storage|energy\s+storage\s+system)\b",
    re.IGNORECASE,
)
_ACOUSTIC_PATTERN = re.compile(
    r"\b(?:noise|acoustic|sound\s+(?:pressure|power)|"
    r"dB(?:[A-C]|\s*\([A-C]\))?|octave[- ]band|attenuation|insertion\s+loss)"
    r"(?![A-Za-z])",
    re.IGNORECASE,
)
_ASSESSMENT_PATTERN = re.compile(
    r"\b(?:"
    r"(?:preliminary\s+)?noise\s+(?:(?:impact|environmental|planning)\s+)?assessment|"
    r"environmental\s+noise\s+assessment|"
    r"noise\s+(?:(?:impact|environmental)\s+)?(?:analysis|report|study)|"
    r"acoustic\s+(?:impact\s+)?assessment(?:\s+report)?|"
    r"sound\s+level\s+assessment(?:\s+report)?|"
    r"(?:assessment|analysis)\s+report"
    r")\b",
    re.IGNORECASE,
)
_BESS_ACOUSTIC_SECTION_PATTERN = re.compile(
    r"\b(?:BESS|battery\s+energy\s+storage\s+system)\s+"
    r"(?:(?:operational|facility|project)\s+)?(?:noise|sound|acoustic)\b",
    re.IGNORECASE,
)
_MANUFACTURER_CITATION_PATTERN = re.compile(
    r"\bmanufacturer(?:'s)?\s+(?:data|specification|specifications|"
    r"sound|noise|rating|rated)\b|\bper\s+manufacturer\s+specification\b",
    re.IGNORECASE,
)
_TECHNICAL_DOCUMENT_PATTERN = re.compile(
    r"\b(?:technical\s+)?(?:data\s+sheet|datasheet|product\s+manual|"
    r"installation\s+manual|product\s+specification|technical\s+specification)\b",
    re.IGNORECASE,
)
_MANUFACTURER_ISSUER_PATTERN = re.compile(
    r"\b(?:manufactured|issued|published|prepared)\s+by\b|\bmanufacturer\b",
    re.IGNORECASE,
)
_RESEARCH_PATTERN = re.compile(
    r"\b(?:proceedings|conference|journal|peer[- ]reviewed|research\s+paper|"
    r"academic\s+research|abstract)\b",
    re.IGNORECASE,
)
_CROSS_INDUSTRY_PATTERN = re.compile(
    r"\b(?:wind\s+turbine|wind\s+farm|BS\s*8233|residential\s+development|"
    r"building\s+(?:acoustics|insulation)|compressor|railway|mining|"
    r"industrial\s+plant|factory|data\s+cent(?:er|re))\b",
    re.IGNORECASE,
)


def classify_pages(
    pages: Iterable[ExtractedPage],
    parse_review_flags: Iterable[str] = (),
    document_title: str = "",
) -> tuple[DocumentClassification, ...]:
    page_list = tuple(pages)
    full_text = "\n".join(page.text for page in page_list)
    front_text = "\n".join(page.text for page in page_list[:3])
    acoustic_present = bool(_ACOUSTIC_PATTERN.search(full_text))
    assessment_match = _ASSESSMENT_PATTERN.search(full_text)
    front_assessment = bool(_ASSESSMENT_PATTERN.search(front_text))
    bess_pages = tuple(
        page for page in page_list if _BESS_PATTERN.search(page.text)
    )
    title_bess = bool(_BESS_PATTERN.search(document_title))
    front_bess = bool(_BESS_PATTERN.search(front_text))
    dedicated_bess_section = bool(_BESS_ACOUSTIC_SECTION_PATTERN.search(full_text))
    assessment_strength = (
        1.0 if front_assessment else 0.72 if assessment_match else 0.0
    )
    direct_focus_strength = (
        1.0
        if title_bess
        else 0.82
        if dedicated_bess_section and assessment_match
        else 0.68
        if len(bess_pages) >= 3 and assessment_match and acoustic_present
        else 0.0
    )
    direct_bess = direct_focus_strength > 0 and acoustic_present
    cross_industry = bool(
        _CROSS_INDUSTRY_PATTERN.search(document_title + "\n" + full_text)
    )
    manufacturer_citation = bool(_MANUFACTURER_CITATION_PATTERN.search(full_text))
    research_signal = bool(
        _RESEARCH_PATTERN.search(document_title + "\n" + "\n".join(
            page.text for page in page_list[:5]
        ))
    )
    manufacturer_issued = bool(
        _TECHNICAL_DOCUMENT_PATTERN.search(document_title + "\n" + front_text)
        and _MANUFACTURER_ISSUER_PATTERN.search(front_text)
        and not assessment_match
    )
    all_review_flags = set(parse_review_flags)
    all_review_flags.update(flag for page in page_list for flag in page.review_flags)
    incidental_bess_mention = bool(bess_pages) and not direct_bess
    needs_review = bool(all_review_flags or incidental_bess_mention)
    available_pages = tuple(page for page in page_list if page.text.strip())
    if not available_pages:
        return (
            DocumentClassification(
                category="relevance_unassessed",
                score=0.0,
                direct_bess_evidence=False,
                cross_industry=False,
                is_primary=True,
                review_status="needs_review",
                evidence=(),
            ),
        )

    results: list[DocumentClassification] = []
    acoustic_strength = 1.0 if _ACOUSTIC_PATTERN.search(front_text) else 0.72 if acoustic_present else 0.0
    bess_coverage = min(len(bess_pages) / 8, 1.0)
    focus_evidence = _collect_evidence(available_pages, (_BESS_PATTERN,))
    assessment_evidence = _collect_evidence(available_pages, (_ASSESSMENT_PATTERN,))
    acoustic_evidence = _collect_evidence(available_pages, (_ACOUSTIC_PATTERN,))
    citation_evidence = _collect_evidence(
        available_pages,
        (_MANUFACTURER_CITATION_PATTERN,),
    )
    research_evidence = _collect_evidence(available_pages, (_RESEARCH_PATTERN,))
    cross_evidence = _collect_evidence(
        available_pages,
        (_CROSS_INDUSTRY_PATTERN,),
    )
    title_type = _assessment_specificity(front_text, document_title)

    def add(
        category: str,
        score: float,
        evidence: list[ClassificationEvidence],
        *,
        category_cross_industry: bool = False,
    ) -> None:
        results.append(
            DocumentClassification(
                category=category,
                score=round(min(score, 0.94), 2),
                direct_bess_evidence=direct_bess,
                cross_industry=category_cross_industry,
                is_primary=False,
                review_status="needs_review" if needs_review else "unreviewed",
                evidence=tuple(_deduplicate_evidence(evidence)),
            )
        )

    if direct_bess and assessment_match:
        impact_score = (
            0.12
            + 0.20 * direct_focus_strength
            + 0.18 * assessment_strength
            + 0.15 * acoustic_strength
            + 0.20 * title_type
            + 0.08 * min(len(bess_pages) / 5, 1.0)
            + 0.07 * float(front_bess)
        )
        add(
            "bess_acoustic_impact_assessment",
            min(impact_score, 0.9),
            focus_evidence + assessment_evidence + acoustic_evidence,
        )

    if manufacturer_issued:
        manufacturer_evidence = _collect_evidence(
            available_pages,
            (_TECHNICAL_DOCUMENT_PATTERN, _MANUFACTURER_ISSUER_PATTERN),
        )
        add(
            "manufacturer_issued_technical_documentation",
            0.32 + 0.18 * float(direct_bess) + 0.14 * acoustic_strength
            + 0.08 * float(front_bess),
            manufacturer_evidence + acoustic_evidence,
        )

    if manufacturer_citation and assessment_match and acoustic_present:
        citation_score = (
            0.25
            + 0.16 * float(manufacturer_citation)
            + 0.13 * assessment_strength
            + 0.12 * direct_focus_strength
            + 0.1 * acoustic_strength
            + 0.06 * min(len(bess_pages) / 5, 1.0)
        )
        add(
            "assessment_cites_manufacturer_specifications",
            citation_score,
            citation_evidence + assessment_evidence + focus_evidence,
        )

    if research_signal and acoustic_present:
        research_score = (
            0.25
            + 0.16 * float(research_signal)
            + 0.12 * direct_focus_strength
            + 0.12 * acoustic_strength
            + 0.08 * bess_coverage
            + 0.06 * min(len(research_evidence) / 4, 1.0)
        )
        add(
            "academic_or_industry_research",
            research_score,
            research_evidence + focus_evidence + acoustic_evidence,
        )

    if not direct_bess and acoustic_present and cross_industry:
        cross_strength = 1.0 if re.search(
            r"\b(?:wind\s+turbine|BS\s*8233|building\s+acoustics)\b",
            document_title + "\n" + front_text,
            re.I,
        ) else 0.72
        cross_score = (
            0.25
            + 0.13 * acoustic_strength
            + 0.15 * cross_strength
            + 0.1 * assessment_strength
            + 0.07 * min(len(cross_evidence) / 4, 1.0)
            + 0.05 * float(incidental_bess_mention)
        )
        add(
            "related_cross_industry_acoustic_technology",
            cross_score,
            cross_evidence + acoustic_evidence + focus_evidence,
            category_cross_industry=True,
        )

    if direct_focus_strength > 0 and not acoustic_present:
        engineering_evidence = _collect_evidence(available_pages, (_BESS_PATTERN,))
        add(
            "general_bess_engineering_documentation",
            0.28 + 0.2 * direct_focus_strength + 0.12 * bess_coverage,
            engineering_evidence,
        )

    if not results:
        results.append(
            DocumentClassification(
                category="unrelated_or_irrelevant",
                score=0.08,
                direct_bess_evidence=direct_bess,
                cross_industry=False,
                is_primary=True,
                review_status="needs_review",
                evidence=(
                    ClassificationEvidence(
                        page_number=available_pages[0].page_number,
                        matched_term="no_category_signal",
                        excerpt="No configured BESS or acoustic category signals were found in extracted text.",
                        evidence_status="not_found",
                    ),
                ),
            )
        )
    else:
        priority = {
            "bess_acoustic_impact_assessment": 7,
            "manufacturer_issued_technical_documentation": 6,
            "assessment_cites_manufacturer_specifications": 5,
            "academic_or_industry_research": 4,
            "related_cross_industry_acoustic_technology": 3,
            "general_bess_engineering_documentation": 2,
        }
        primary_index = max(
            range(len(results)),
            key=lambda index: (priority[results[index].category], results[index].score),
        )
        results[primary_index] = replace(results[primary_index], is_primary=True)
    return tuple(results)


def _assessment_specificity(front_text: str, document_title: str) -> float:
    source = document_title + "\n" + front_text
    if re.search(r"\bnoise\s+impact\s+assessment\b", source, re.I):
        return 1.0
    if re.search(r"\bpreliminary\s+noise\s+assessment\b", source, re.I):
        return 0.94
    if re.search(r"\b(?:sound\s+level|acoustic)\s+assessment\s+report\b", source, re.I):
        return 0.86
    if re.search(r"\bnoise\s+analysis\s+report\b", source, re.I):
        return 0.82
    if re.search(r"\bnoise\s+report\b", source, re.I):
        return 0.74
    return 0.68


def _collect_evidence(
    pages: tuple[ExtractedPage, ...],
    patterns: tuple[re.Pattern[str], ...],
) -> list[ClassificationEvidence]:
    evidence: list[ClassificationEvidence] = []
    seen: set[tuple[int, str]] = set()
    for page in pages:
        for pattern in patterns:
            for match in list(pattern.finditer(page.text))[:3]:
                term = match.group(0).strip()
                key = (page.page_number, term.casefold())
                if key in seen:
                    continue
                seen.add(key)
                start = max(0, match.start() - 90)
                end = min(len(page.text), match.end() + 90)
                excerpt = " ".join(page.text[start:end].split())
                evidence.append(
                    ClassificationEvidence(
                        page_number=page.page_number,
                        matched_term=term,
                        excerpt=excerpt,
                    )
                )
    return evidence


def _deduplicate_evidence(
    evidence: list[ClassificationEvidence],
) -> list[ClassificationEvidence]:
    output: list[ClassificationEvidence] = []
    seen: set[tuple[int, str, str]] = set()
    for item in evidence:
        key = (item.page_number, item.matched_term.casefold(), item.excerpt)
        if key not in seen:
            output.append(item)
            seen.add(key)
    return output
