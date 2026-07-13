# ScriptOCR

A document OCR and field extraction system that converts handwritten and printed documents into structured data, automatically pushed into Google Sheets. Built as a portfolio project inspired by real document workflows at NJV High School, Karachi.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://img.shields.io/badge/Python-3.11-blue) [![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://img.shields.io/badge/FastAPI-0.100+-green) [![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)](https://img.shields.io/badge/OpenCV-4.x-red) [![GPT-4o](https://img.shields.io/badge/GPT--4o-Vision-orange)](https://img.shields.io/badge/GPT--4o-Vision-orange)

---

## What it does

Upload an image of any handwritten or printed document — one at a time, a batch of independent documents, or multiple pages of the same document. ScriptOCR preprocesses the image, extracts all text using GPT-4o vision (or Gemini, see Configuration), intelligently maps the extracted text to the specific fields of the document type, and pushes the structured data straight into Google Sheets — returning clean JSON ready for further processing or integration.

---

## Features

- **Image Preprocessing** — Grayscale conversion, denoising, deskewing, and contrast enhancement via OpenCV
- **Handwriting OCR** — Powered by GPT-4o vision (GitHub Models) or Gemini, switchable via one environment variable
- **Field Extraction** — Intelligent field mapping per document type using structured prompts, done in the same model call as OCR for single/batch documents
- **Data-Driven Templates** — Document schemas live in `templates.json`; adding a new document type is a JSON edit, not a code change
- **Pluggable Export Layer** — Google Sheets today, other destinations addable without touching the API layer
- **Google Sheets Integration** — Every successful extraction is auto-pushed as a new row, with a UI confirmation showing the row number
- **Multi-Image Batch Upload** — Process up to 15 independent documents in one request, with bounded concurrency and automatic retry on transient rate limits
- **Multi-Page Merge Mode** — Combine multiple page images of ONE document into a single extraction and a single exported row
- **REST API** — FastAPI backend with endpoints for single-document, batch, and merge OCR + field extraction
- **Web UI** — Clean, professional interface supporting single-file, batch, and merge workflows
- **Confidence Scoring** — Each extraction returns a confidence level and flags uncertain words

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Image Processing | OpenCV |
| OCR Engine | GPT-4o (GitHub Models) or Gemini — configurable |
| Backend | FastAPI, Uvicorn |
| Sheets Integration | gspread, google-auth |
| Frontend | HTML, CSS, JavaScript |

---

## Project Structure

```
ScriptOCR/
├── api.py             # FastAPI backend — shared pipeline, single/batch/merge endpoints
├── field_mapper.py    # Document field extraction logic (reads templates.py)
├── llm_config.py       # Single source of truth for LLM provider/model config
├── templates.json      # Document schemas + export config (data, not code)
├── templates.py        # Template store loader
├── exporters.py         # Pluggable export destination layer (Google Sheets today)
├── index.html           # Web UI (single-file, batch, and merge upload)
├── requirements.txt      # Pinned dependencies
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Either a GitHub account (free GPT-4o access via GitHub Models) or a Google account (free Gemini API key)
- A Google Cloud service account (for Sheets export)

### Installation

```
git clone https://github.com/Qasim-Bukhari/ScriptOCR.git
cd ScriptOCR

python -m venv venv
venv\Scripts\activate        # Windows — Mac/Linux: source venv/bin/activate

python -m pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root (already gitignored — never commit this file):

```
GITHUB_TOKEN=your_github_token_here
LLM_PROVIDER=github
ALLOWED_ORIGINS=http://127.0.0.1:8000,http://localhost:8000
```

To get a free GitHub token with GPT-4o access:
1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
2. Generate a new token (classic) with default scopes
3. Use it to access [GitHub Models](https://github.com/marketplace/models)

**Switching to Gemini instead** (e.g. if GitHub Models' daily quota runs out): set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY=your_key` in `.env` instead. See `llm_config.py` for details — no code changes needed either way.

For Google Sheets export, place a Google Service Account key as `credentials.json` in the project root (gitignored, not included in this repo). Share each target Sheet with that service account's `client_email` as an Editor.

### Run the server

```
uvicorn api:app --reload
```

Open the web UI at:
```
http://127.0.0.1:8000/ui
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| GET | `/document-types` | List available document types |
| POST | `/ocr` | Extract raw text from a single image |
| POST | `/ocr/fields` | Extract and map fields from a single image, push to Sheets |
| POST | `/ocr/fields/batch` | Same pipeline for up to 15 independent documents at once |
| POST | `/ocr/fields/merge` | Merge up to 15 page images of ONE document into a single extraction + row |

### Example Request (single document)

```
curl -X POST http://127.0.0.1:8000/ocr/fields \
  -F "file=@document.jpg" \
  -F "document_type=leave_request"
```

### Example Response

```json
{
  "success": true,
  "filename": "document.jpg",
  "document_type": "leave_request",
  "document_name": "Leave / Absence Request Form",
  "extracted_text": "Leave / Absence Request Form\nDate: 04 July 2026\n...",
  "fields": {
    "date": "04 July 2026",
    "employee_name": "Hamza Tariq",
    "department": "IT Support",
    "leave_type": "Casual",
    "start_date": "10 July 2026",
    "end_date": "10 July 2026",
    "number_of_days": "1",
    "reason": "Personal error - visiting NADRA office for CNIC Renewal.",
    "approved_by": "Nadia Chaudhry"
  },
  "low_confidence_words": [],
  "confidence": "high",
  "sheet": {
    "success": true,
    "row_number": 50
  }
}
```

### Example Request (batch — independent documents)

```
curl -X POST http://127.0.0.1:8000/ocr/fields/batch \
  -F "files=@doc1.jpg" \
  -F "files=@doc2.jpg" \
  -F "document_type=procurement_request"
```

Returns a `summary` (`total`/`succeeded`/`failed`/`saved`) plus a `results` array with one entry per image — a failure on one document never blocks the rest of the batch.

### Example Request (merge — pages of one document)

```
curl -X POST http://127.0.0.1:8000/ocr/fields/merge \
  -F "files=@page1.jpg" \
  -F "files=@page2.jpg" \
  -F "document_type=employee_registration_full"
```

Unlike batch mode, a failure on any page fails the whole request — a partial multi-page form isn't a usable record.

---

## Supported Document Types

| Key | Document |
|---|---|
| `leave_request` | Leave / Absence Request Form |
| `procurement_request` | Procurement / Inventory Request Form |
| `employee_registration` | Employee Registration Form (Personal Info — Page 1 only) |
| `employee_registration_full` | Employee Registration Form (Full — Personal + Employment, 2 pages) |

New document types are added by editing `templates.json` — no Python changes required.

---

## Roadmap

- [x] CLI OCR Engine
- [x] FastAPI Backend
- [x] Field Mapper
- [x] Google Sheets Integration
- [x] Web UI
- [x] Data-Driven Templates & Pluggable Exporters
- [x] Multi-Image Batch Upload
- [x] Multi-Page Merge Mode
- [x] Multi-provider LLM support (GitHub Models / Gemini)
- [ ] Auto document type detection
- [ ] More NJV department schemas (real data collection)
- [ ] Server deployment
- [ ] Self-hosted fine-tuned OCR model
- [ ] Processing history and logs

---

## Use Case

Inspired by real document workflows at **NJV High School, Karachi**, where multiple departments handle handwritten forms manually entered into external software. ScriptOCR automates that process — from image capture to structured data landing in a spreadsheet.

---

## Author

**Qasim Bukhari**
CS Graduate & Software Engineer | STEAM Educator, Karachi