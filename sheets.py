"""
ScriptOCR — Google Sheets Integration (Phase 4)

DEPRECATED as of the Exporter refactor (Phase 7 groundwork).
push_to_sheet() is kept only so nothing that still imports it breaks.
New code should use exporters.get_exporter() / exporters.GoogleSheetsExporter
directly — the real logic now lives in exporters.py.

Usage: python sheets.py
"""

from exporters import GoogleSheetsExporter


def push_to_sheet(document_type: str, fields: dict) -> dict:
    return GoogleSheetsExporter().export(document_type, fields)


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
        "address": "Garden, Karachi",
    }

    print("\n Testing Google Sheets integration...")
    result = push_to_sheet("student_registration", sample_fields)
    if result["success"]:
        print(f" ✓ Data pushed successfully to row {result['row_number']}")
        print(f" ✓ Check your sheet: https://docs.google.com/spreadsheets/d/{result['sheet_id']}")
    else:
        print(" ✗ Failed to push data")
