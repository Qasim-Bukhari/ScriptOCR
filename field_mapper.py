"""
Handwriting OCR — Phase 3 (refactored on top of the Phase 7 template store)
Field Mapper

Extracts structured fields from raw OCR text based on document type.
Document schemas now live in templates.json — see templates.py.

Usage: python field_mapper.py
"""

import os
import json
from openai import OpenAI

from templates import get_template, TEMPLATES, TemplateNotFoundError

# ── Configuration ─────────────────────────────────────────────────────────────
# Shared with api.py via llm_config.py — see that file for how to switch
# providers (e.g. when a daily free-tier quota runs out).
from llm_config import LLM_API_KEY as GITHUB_TOKEN, LLM_ENDPOINT as GITHUB_ENDPOINT, LLM_MODEL as MODEL

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


def map_fields(document_type: str, extracted_text: str) -> dict:
    # get_template() raises TemplateNotFoundError for unknown document types —
    # api.py catches this and turns it into a 400 response.
    get_template(document_type)

    client = OpenAI(base_url=GITHUB_ENDPOINT, api_key=GITHUB_TOKEN)
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
    Date: 29 June 2026
    Requested By: Qasim Ahmed
    Department: Science Room
    Item: Whiteboard Markers
    Quantity: 10 pieces
    Purpose: Required for daily classroom teaching
    Signature: Q. Ahmed
    """

    document_type = "store_request"
    print(f"\n Testing Field Mapper with document type: {document_type}")
    print(f" Input text:\n{sample_text}")

    result = map_fields(document_type, sample_text)
    display_result(document_type, result)