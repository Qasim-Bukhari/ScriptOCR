"""
Handwriting OCR — Phase 3 (refactored on top of the Phase 7 template store)
Field Mapper

Extracts structured fields from raw OCR text based on document type.
Document schemas now live in templates.json — see templates.py.

Usage: python field_mapper.py
"""

import os
import json
import base64
from openai import OpenAI

from templates import get_template, TEMPLATES, TemplateNotFoundError

# ── Configuration ─────────────────────────────────────────────────────────────
# Shared with api.py via llm_config.py — see that file for how to switch
# providers (e.g. when a daily free-tier quota runs out).
from llm_config import LLM_MODEL as MODEL, llm_client

# Backward-compatible alias. The source of truth is now templates.json,
# but anything that still imports DOCUMENT_SCHEMAS from here keeps working.
DOCUMENT_SCHEMAS = TEMPLATES


# ── Field Mapper ───────────────────────────────────────────────────────────────
def build_prompt(document_type: str, extracted_text: str) -> str:
    schema = get_template(document_type)

    fields_description = "\n".join(
        f'- "{field}": {description}'
        for field, description in schema["fields"].items()
    )
    empty_fields = {field: "" for field in schema["fields"]}

    return f"""You are an intelligent document field extractor.

You will be given raw text extracted from a handwritten "{schema['name']}".
Your job is to identify and extract the value for each field listed below.

Fields to extract:
{fields_description}

Rules:
1. Extract only what is clearly present in the text.
2. If a field is not found, leave it as an empty string "".
3. Do not guess or invent values.
4. Return ONLY a JSON object with these exact field names.
5. No markdown, no backticks, no explanation.
6. For any field that represents a date, normalize it to "DD Month YYYY"
   (e.g. "04 July 2026") — no hyphens, no slashes, no extra spaces around
   words, regardless of how it was written or punctuated in the source text.

Expected output format:
{json.dumps(empty_fields, indent=2)}

Raw OCR text to process:
\"\"\"{extracted_text}\"\"\"
"""


# ── Combined OCR + Field Mapping (single model call) ───────────────────────────
def build_combined_prompt(document_type: str) -> str:
    """Same field schema as build_prompt(), but framed as a combined
    OCR-and-extract task performed directly against an image, rather than
    against already-extracted text."""
    schema = get_template(document_type)

    fields_description = "\n".join(
        f'- "{field}": {description}'
        for field, description in schema["fields"].items()
    )
    empty_fields = {field: "" for field in schema["fields"]}

    return f"""You are an expert OCR and document field extraction system, specialized in handwritten and cursive English text.

You will be shown an image of a handwritten "{schema['name']}". In a SINGLE pass, do BOTH of the following steps.

STEP 1 — Extract all handwritten text from the image exactly as written, preserving line breaks, bullet points, and indentation. Pay close attention to connected cursive script and symbols (like / or -). Wrap any word you are genuinely uncertain about in [?word?] markers.

STEP 2 — From that same text, identify and extract the value for each of these fields:
{fields_description}

Field extraction rules:
- Extract only what is clearly present in the text.
- If a field is not found, leave it as an empty string "".
- Do not guess or invent values.
- For any field that represents a date, normalize it to "DD Month YYYY" (e.g. "04 July 2026") — no hyphens, no slashes, no extra spaces, regardless of how it was written or punctuated in the source.

Return ONLY a single JSON object in exactly this format — no markdown, no backticks, no explanation:
{{
  "extracted_text": "the full extracted text here",
  "low_confidence_words": ["word1", "word2"],
  "confidence": "high | medium | low",
  "fields": {json.dumps(empty_fields, indent=2)}
}}
"""


def extract_and_map_fields(image_path: str, document_type: str) -> dict:
    """Does OCR + field mapping in ONE model call instead of two sequential
    round-trips (extract_text() then map_fields()). This roughly halves
    both per-document latency and API-quota usage — the single biggest
    lever for making the app feel responsive in a live demo.

    Used by the single-document and batch pipelines, where one image maps
    to one document. NOT used by merge mode's per-page OCR step — a
    multi-page document's fields can span several images, so those pages
    still go through OCR-only extraction (see ocr_only() in api.py), with
    ONE separate map_fields() call afterwards on the combined text from
    all pages. That's still just one field-mapping call per document
    either way, so merge mode was already efficient here; this change is
    what speeds up single-document and batch mode specifically.
    """
    get_template(document_type)  # raises TemplateNotFoundError early for unknown types

    client = llm_client
    prompt = build_combined_prompt(document_type)

    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
        ]}]
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
    if raw.startswith("json"):
        raw = raw[4:]

    return json.loads(raw.strip())


def map_fields(document_type: str, extracted_text: str) -> dict:
    # get_template() raises TemplateNotFoundError for unknown document types —
    # api.py catches this and turns it into a 400 response.
    get_template(document_type)

    client = llm_client
    prompt = build_prompt(document_type, extracted_text)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
    if raw.startswith("json"):
        raw = raw[4:]

    return json.loads(raw.strip())


# ── Display Result ─────────────────────────────────────────────────────────────
def display_result(document_type: str, fields: dict):
    schema_name = get_template(document_type)["name"]
    print("\n" + "═" * 60)
    print(f"  EXTRACTED FIELDS — {schema_name}")
    print("═" * 60)
    for field, value in fields.items():
        label = field.replace("_", " ").title()
        value_display = value if value else "—"
        print(f"  {label:<20}: {value_display}")
    print("═" * 60 + "\n")


# ── Main (Test) ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_text = """
    Date: 04 July 2026
    Employee Name: Ayesha Raza
    Department: Finance & Accounts
    Leave Type: Sick
    Start Date: 06 July 2026
    End Date: 08 July 2026
    Number of Days: 3
    Reason: Recovering from viral fever, advised bed rest by physician.
    Approved By: Zainab Hussain
    """

    document_type = "leave_request"
    print(f"\n Testing Field Mapper with document type: {document_type}")
    print(f" Input text:\n{sample_text}")

    result = map_fields(document_type, sample_text)
    display_result(document_type, result)