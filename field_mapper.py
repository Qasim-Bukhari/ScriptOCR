"""
Handwriting OCR — Phase 3
Field Mapper
Extracts structured fields from raw OCR text based on document type.
Usage: python field_mapper.py
"""

import os
import json
from openai import OpenAI


# ── Configuration ─────────────────────────────────────────────────────────────

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_ENDPOINT = "https://models.inference.ai.azure.com"
MODEL           = "gpt-4o"

# ── Document Schemas ───────────────────────────────────────────────────────────

DOCUMENT_SCHEMAS = {
    "store_request": {
        "name": "Store Request Form",
        "fields": {
            "date": "The date the request was made",
            "requested_by": "Full name of the person making the request",
            "department": "Department or section of the requester",
            "item_name": "Name of the item being requested",
            "quantity": "Number or amount of items requested",
            "purpose": "Reason or purpose for the request",
            "signature": "Signature or signatory name"
        }
    },
    "student_registration": {
        "name": "Student Registration Form",
        "fields": {
            "date": "Date of registration",
            "student_name": "Full name of the student",
            "father_name": "Father's full name",
            "date_of_birth": "Student's date of birth",
            "class": "Class or grade the student is enrolling in",
            "section": "Section assigned to the student",
            "roll_number": "Roll number assigned",
            "contact_number": "Contact phone number",
            "address": "Home address of the student"
        }
    },
    "procurement_request": {
        "name": "Procurement / Inventory Request Form",
        "fields": {
            "date": "Date of the request",
            "requested_by": "Name of the person requesting",
            "department": "Department making the request",
            "item_name": "Name of the item or equipment",
            "quantity": "Quantity required",
            "estimated_cost": "Estimated cost if mentioned",
            "purpose": "Purpose or justification for procurement",
            "approved_by": "Name of approving authority if mentioned"
        }
    }
}


# ── Field Mapper ───────────────────────────────────────────────────────────────

def build_prompt(document_type: str, extracted_text: str) -> str:
    schema = DOCUMENT_SCHEMAS[document_type]
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

Expected output format:
{json.dumps(empty_fields, indent=2)}

Raw OCR text to process:
\"\"\"{extracted_text}\"\"\"
"""


def map_fields(document_type: str, extracted_text: str) -> dict:
    if document_type not in DOCUMENT_SCHEMAS:
        raise ValueError(f"Unknown document type: {document_type}. "
                         f"Available: {list(DOCUMENT_SCHEMAS.keys())}")

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
    schema_name = DOCUMENT_SCHEMAS[document_type]["name"]
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

    # Sample OCR output to test the field mapper
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

    print(f"\n  Testing Field Mapper with document type: {document_type}")
    print(f"  Input text:\n{sample_text}")

    result = map_fields(document_type, sample_text)
    display_result(document_type, result)
