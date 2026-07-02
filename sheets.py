"""
ScriptOCR — Google Sheets Integration
Phase 4: Push extracted fields directly into Google Sheets
"""

import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime


# ── Configuration ─────────────────────────────────────────────────────────────

CREDENTIALS_FILE = "credentials.json"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# Sheet IDs per document type
SHEET_IDS = {
    "student_registration": "1yO_iKntsAWuMdOLjRptGgWCvvBC9nLPkni9cJ2JCFik"
}

# Column order per document type — must match sheet headers exactly
SHEET_COLUMNS = {
    "student_registration": [
        "date",
        "student_name",
        "father_name",
        "date_of_birth",
        "class",
        "section",
        "roll_number",
        "contact_number",
        "address"
    ]
}


# ── Google Sheets Client ───────────────────────────────────────────────────────

def get_client():
    creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
    return gspread.authorize(creds)


# ── Push to Sheet ──────────────────────────────────────────────────────────────

def push_to_sheet(document_type: str, fields: dict) -> dict:
    """
    Push extracted fields to the corresponding Google Sheet.
    Returns result with row number and status.
    """
    if document_type not in SHEET_IDS:
        raise ValueError(f"No sheet configured for document type: {document_type}")

    client   = get_client()
    sheet_id = SHEET_IDS[document_type]
    sheet    = client.open_by_key(sheet_id).sheet1

    # Build row in correct column order
    columns  = SHEET_COLUMNS[document_type]
    row_data = [fields.get(col, "") for col in columns]

    # Append timestamp
    row_data.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Append row to sheet
    sheet.append_row(row_data)

    # Get the row number that was just added
    row_number = len(sheet.get_all_values())

    return {
        "success": True,
        "sheet_id": sheet_id,
        "row_number": row_number,
        "data_pushed": dict(zip(columns, row_data))
    }


# ── Test ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_fields = {
        "date": "01 July 2026",
        "student_name": "Qasim Bukhari",
        "father_name": "Aslam Bukhari",
        "date_of_birth": "21/09/2003",
        "class": "12",
        "section": "B",
        "roll_number": "123",
        "contact_number": "0300-1234567",
        "address": "Garden, Karachi"
    }

    print("\n  Testing Google Sheets integration...")
    result = push_to_sheet("student_registration", sample_fields)

    if result["success"]:
        print(f"  ✓ Data pushed successfully to row {result['row_number']}")
        print(f"  ✓ Check your sheet: https://docs.google.com/spreadsheets/d/{result['sheet_id']}")
    else:
        print("  ✗ Failed to push data")
