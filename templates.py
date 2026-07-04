"""
ScriptOCR — Template Store

Document schemas used to live hardcoded as DOCUMENT_SCHEMAS inside
field_mapper.py. They now live in templates.json (data), loaded here.

Adding a new document type is now a JSON edit, not a code change —
no redeploy required. This is the Phase 7 foundation: "visit a
department, fill out a template" instead of "visit a department,
write Python."

Each template entry looks like:
{
    "name": "Display Name",
    "fields": { "field_key": "description for the AI prompt", ... },
    "export": {                      # optional — omit if no destination yet
        "type": "google_sheets",
        "sheet_id": "...",
        "columns": ["field_key", ...]
    }
}
"""

import json
import os

TEMPLATES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates.json")


class TemplateNotFoundError(Exception):
    """Raised when a document_type has no matching template."""
    pass


def _load_templates() -> dict:
    with open(TEMPLATES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# Loaded once at import time.
TEMPLATES = _load_templates()


def reload_templates() -> None:
    """Re-reads templates.json from disk. Useful for admin tooling later
    (add a template via UI, call this instead of restarting the server)."""
    global TEMPLATES
    TEMPLATES = _load_templates()


def list_templates() -> dict:
    """Returns {document_type: display_name} for every known template."""
    return {key: value["name"] for key, value in TEMPLATES.items()}


def get_template(document_type: str) -> dict:
    """Returns the full template dict for a document type.
    Raises TemplateNotFoundError if the document_type is unknown."""
    if document_type not in TEMPLATES:
        raise TemplateNotFoundError(
            f"Unknown document type: '{document_type}'. "
            f"Available: {list(TEMPLATES.keys())}"
        )
    return TEMPLATES[document_type]


def get_export_config(document_type: str):
    """Returns the 'export' block for a template, or None if this
    document type has no export destination configured yet."""
    return get_template(document_type).get("export")
