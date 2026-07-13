"""
ScriptOCR — Exporter Layer

Extraction and delivery are now decoupled. api.py no longer knows
anything about Google Sheets specifically — it asks templates.py
"where does this document type export to?" and gets back an Exporter
it can call generically.

Adding a new destination (Excel, a webhook, a database) means adding
one new Exporter subclass and registering it below. api.py and
field_mapper.py never need to change.
"""

import os
from abc import ABC, abstractmethod
from datetime import datetime
import re

import gspread
from google.oauth2.service_account import Credentials

from templates import get_export_config


def _normalize_header(text: str) -> str:
    """Turns a human header like 'Employee Name' or 'S. No.' into a
    lookup key like 'employee_name' or 's_no', so it can be matched
    against field keys (which are already snake_case) regardless of
    capitalization, punctuation, or spacing in the sheet."""
    text = text.strip().lower()
    text = re.sub(r"[^\w]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


# Header names that don't correspond to an extracted field, but that the
# exporter still knows how to fill in automatically. Add more aliases here
# as needed — no code elsewhere needs to change.
_AUTO_SERIAL_HEADERS = {"s_no", "sr_no", "sno", "serial_no", "serial_number", "no"}
_AUTO_TIMESTAMP_HEADERS = {"timestamp", "submitted_at", "date_submitted", "submitted_on"}


# ── Row-count cache ──────────────────────────────────────────────────────
# sheet_id -> last known row count (header + data rows), equivalent to what
# `len(sheet.get_all_values())` used to return on every single export.
#
# Seeded once per sheet per server run via a cheap single-column read
# (col_values(1)), then kept in sync from each append — one row added per
# successful export, so the cache just increments by 1 instead of doing
# another read. This replaces a full-grid read on every export, which was
# also happening while holding api.py's global _export_lock — meaning it
# was serializing ALL exports (even to unrelated sheets) behind an
# increasingly expensive read as any one sheet grew.
#
# Safe without its own locking: every export() call already runs behind
# that same global _export_lock in api.py, so exports never happen
# concurrently within this process.
#
# Tradeoff: if a Sheet's rows are added/removed by hand (outside this app)
# while the server keeps running, this cache can drift until the next
# server restart, which reseeds it fresh. Acceptable for a dev/demo
# workflow with frequent restarts. If NJV staff will be actively
# hand-editing these Sheets during long-running sessions, this should
# re-verify periodically instead of only seeding once at startup — flag
# it if that becomes the actual usage pattern.
_row_count_cache: dict[str, int] = {}


def _get_row_count(sheet, sheet_id: str) -> int:
    """Returns the current row count (header + data), using the in-memory
    cache when available. On a cache miss, seeds it with a single-column
    read instead of a full-grid read — cheap regardless of how many
    columns the sheet has, at the cost of assuming column A has no gaps
    (true for every template's export config today, since column A is
    always either a real field or an auto-filled serial/timestamp)."""
    if sheet_id not in _row_count_cache:
        _row_count_cache[sheet_id] = len(sheet.col_values(1))
    return _row_count_cache[sheet_id]


class Exporter(ABC):
    """Base interface every export destination implements."""

    @abstractmethod
    def export(self, document_type: str, fields: dict) -> dict:
        """Push extracted fields to the destination.
        Returns a result dict with at least a 'success' key."""
        raise NotImplementedError


class GoogleSheetsExporter(Exporter):
    """Pushes extracted fields to the Google Sheet configured in this
    document type's template (see templates.json -> "export")."""

    # Configurable so a deployed host can mount the service account key
    # wherever it lands (e.g. Render's Secret Files feature writes it to
    # a path the platform controls) without changing code — defaults to
    # the local dev convention of a credentials.json in the project root.
    CREDENTIALS_FILE = os.environ.get("GOOGLE_CREDENTIALS_PATH", "credentials.json")
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    def _get_client(self):
        creds = Credentials.from_service_account_file(self.CREDENTIALS_FILE, scopes=self.SCOPES)
        return gspread.authorize(creds)

    def export(self, document_type: str, fields: dict) -> dict:
        config = get_export_config(document_type)
        if not config or config.get("type") != "google_sheets":
            raise ValueError(
                f"No Google Sheets export configured for document type: {document_type}"
            )

        sheet_id = config["sheet_id"]
        fallback_columns = config["columns"]

        client = self._get_client()
        sheet = client.open_by_key(sheet_id).sheet1

        header_row = sheet.row_values(1)
        existing_rows = _get_row_count(sheet, sheet_id)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if header_row:
            # Header-driven: match each sheet column by its header text,
            # in whatever order/spacing/capitalization it's actually in.
            # Unknown headers get a blank rather than crashing, so an
            # extra note column etc. doesn't break the export.
            row_data = []
            for header in header_row:
                key = _normalize_header(header)
                if key in fields:
                    row_data.append(fields.get(key, ""))
                elif key in _AUTO_SERIAL_HEADERS:
                    row_data.append(existing_rows)  # header row + prior data rows = next serial number
                elif key in _AUTO_TIMESTAMP_HEADERS:
                    row_data.append(now_str)
                else:
                    row_data.append("")
            columns_used = header_row
        else:
            # No header row yet — fall back to templates.json's declared
            # order (the original behavior), plus a trailing timestamp.
            row_data = [fields.get(col, "") for col in fallback_columns]
            row_data.append(now_str)
            columns_used = fallback_columns + ["timestamp"]

        response = sheet.append_row(row_data, insert_data_option="INSERT_ROWS")
        updated_range = response.get("updates", {}).get("updatedRange", "")
        match = re.search(r"![A-Za-z]+(\d+)", updated_range)
        row_number = int(match.group(1)) if match else None

        # Keep the cache in sync with the row we just added — we know
        # exactly one row was appended, so no extra read is needed.
        _row_count_cache[sheet_id] = existing_rows + 1

        return {
            "success": True,
            "sheet_id": sheet_id,
            "row_number": row_number,
            "data_pushed": dict(zip(columns_used, row_data)),
        }


# ── Exporter Registry ───────────────────────────────────────────────
# Add new export types here as they're built (e.g. "excel": ExcelExporter).
_EXPORTER_REGISTRY = {
    "google_sheets": GoogleSheetsExporter,
}


def get_exporter(document_type: str):
    """Returns an Exporter instance for this document type's configured
    export destination, or None if no export is configured yet."""
    config = get_export_config(document_type)
    if not config:
        return None
    exporter_cls = _EXPORTER_REGISTRY.get(config.get("type"))
    if exporter_cls is None:
        raise ValueError(f"Unknown exporter type: {config.get('type')}")
    return exporter_cls()


# ── Test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_fields = {
        "date": "01 July 2026",
        "employee_name": "Qasim Bukhari",
        "department": "Operations",
        "leave_type": "Casual",
        "start_date": "01 July 2026",
        "end_date": "03 July 2026",
        "number_of_days": "3",
        "reason": "Family event out of town",
        "approved_by": "M. Rehman",
    }

    print("\n Testing Google Sheets exporter...")
    exporter = get_exporter("leave_request")
    result = exporter.export("leave_request", sample_fields)
    if result["success"]:
        print(f" ✓ Data pushed successfully to row {result['row_number']}")
        print(f" ✓ Check your sheet: https://docs.google.com/spreadsheets/d/{result['sheet_id']}")
    else:
        print(" ✗ Failed to push data")