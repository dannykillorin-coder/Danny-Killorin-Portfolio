# Stage B1 benchmark review

## What is stored

`data/private/benchmarks.sqlite3` contains deterministic candidates, append-only review events, and a separate gold-standard table. Candidate payloads retain the source document ID, original URL, filename, SHA-256, page and table, exact source cell text, and parser evidence IDs/coordinates. The original candidate is never updated by a correction. Only an explicit approval creates a gold-standard record.

The Pydantic engineering models distinguish transcription from comparison eligibility. Unknown measurement basis, scope, weighting, units, or operating conditions remain unknown. A comparison-eligible record must have explicit measurement type, units, weighting, basis, aggregation scope, and applicable frequency bandwidth; operating conditions must be reported or explicitly not applicable. Deterministic evidence validation checks the source hash, exact stored quotes and coordinates, numeric value, metric wording, weighting, and parser's header-scoped frequency mapping. It does not treat an LLM score or reviewer confidence as engineering evidence.

## Stage B1.1 candidate generation

Stage B1.1 writes a separate `stage-b1.1` candidate generation; it does not replace the original 50 `stage-b1.0` candidates or their review events. Candidate identity includes the generation version. Its records keep acoustic record kind separate from metric type, distinguish BESS-specific findings from general references, and retain exact numeric comparators and ranges. A missing or ambiguous table reference remains unset rather than borrowing a table number from unrelated nearby prose.

The deterministic pass reads existing parsed pages and tables only. It makes no provider calls and does not approve gold records. Table-alignment problems, unknown semantic fields, and ambiguous cells remain open review items. Review items are persisted idempotently and remain visible on later runs even when no new candidates are added.

Use the CLI to generate and inspect the new version without changing the existing candidate payloads:

```powershell
bai benchmark generate
bai benchmark list --generation-version stage-b1.1
bai benchmark compare
bai benchmark reviews
```

`compare` aligns B1.0 and B1.1 candidates by document, page, table, and source cell. Review each candidate against its original PDF before correcting, rejecting, or approving it. Corrections append review events; they do not overwrite the generated payload. Only manually approved records enter the separate gold-standard table.

## Review flow

1. Generate candidates from existing parsed tables. Generation uses no provider/API.
2. List pending items, then inspect one with `benchmark show`. Compare the displayed page/table and exact quotes against the local PDF.
3. Approve with `--action approve`, reject with `--action reject --reason`, or append a correction with `--action correct`.
4. Export only approved records with `benchmark export`. The JSON export is written under `data/private/`.

The reviewer name is required for every action. Rejections and corrections require a reason. Corrections require a JSON file with this structure. The placeholders below are not research data; replace them with a value and source quote you personally verified:

```json
{
  "measurement": {
    "value": null,
    "source_value_text": "PASTE_EXACT_SOURCE_CELL_TEXT",
    "units": null,
    "metric_type": "unknown"
  },
  "human_review_evidence": [
    {
      "document_id": 17,
      "page_number": 1,
      "table_reference": "PASTE_SOURCE_TABLE_REFERENCE",
      "quote": "PASTE_EXACT_SOURCE_CELL_TEXT",
      "verification_method": "visual_pdf_review"
    }
  ]
}
```

Use the real document ID, page, table, and source wording, and replace `value` with the corrected numeric value when one is supported. `text_match` requires the quote to occur on the cited extracted page. `visual_pdf_review` records that the reviewer checked the original PDF directly when extraction text does not match; the application cannot independently verify that visual attestation. A corrected numeric value and its preserved source text must match the correction evidence. If the source cannot be verified, do not approve the candidate.

An eligible comparison record is not simply a correctly transcribed value. It also needs enough explicit source context to compare like with like. In particular, do not equate sound power with sound pressure, per-unit with facility-total values, predicted with measured values, or unlike weightings/bandwidths.

## Provider eligibility

Provider processing is denied unless an explicit per-document allow event is present. Eligibility changes require both a reviewer and authorization basis and are retained as an event history in the main SQLite database. No external provider is called in Stage B1. Any future provider adapter must check `DocumentStore.is_provider_processing_eligible` before sending document content.
