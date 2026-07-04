# ScriptOCR

A document OCR and field extraction system that converts handwritten and printed documents into structured data, automatically pushed into Google Sheets. Built as a portfolio project inspired by real document workflows at NJV High School, Karachi.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://img.shields.io/badge/Python-3.11-blue) [![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://img.shields.io/badge/FastAPI-0.100+-green) [![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)](https://img.shields.io/badge/OpenCV-4.x-red) [![GPT-4o](https://img.shields.io/badge/GPT--4o-Vision-orange)](https://img.shields.io/badge/GPT--4o-Vision-orange)

---

## What it does

Upload an image of any handwritten or printed document — one at a time, or a whole batch at once. ScriptOCR preprocesses the image, extracts all text using GPT-4o vision, intelligently maps the extracted text to the specific fields of the document type, and pushes the structured data straight into Google Sheets — returning clean JSON ready for further processing or integration.

---

## Features

- **Image Preprocessing** — Grayscale conversion, denoising, deskewing, and contrast enhancement via OpenCV
- **Handwriting OCR** — Powered by GPT-4o vision for high accuracy on cursive and printed text
- **Field Extraction** — Intelligent field mapping per document type using structured prompts
- **Data-Driven Templates** — Document schemas live in `templates.json`; adding a new document type is a JSON edit, not a code change
- **Pluggable Export Layer** — Google Sheets today, other destinations (Excel, webhooks) addable without touching the API layer
- **Google Sheets Integration** — Every successful extraction is auto-pushed as a new row, with a UI confirmation showing the row number
- **Multi-Image Batch Upload** — Process up to 15 documents in one request, with bounded concurrency and automatic retry on transient rate limits
- **REST API** — FastAPI backend with endpoints for single-document and batch OCR + field extraction
- **Web UI** — Clean, professional side-by-side interface supporting both single-file and batch workflows
- **Confidence Scoring** — Each extraction returns a confidence level and flags uncertain words

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Image Processing | OpenCV, Pillow |
| OCR Engine | GPT-4o via GitHub Models |
| Backend | FastAPI, Uvicorn |
| Sheets Integration | gspread, google-auth |
| Frontend | HTML, CSS, JavaScript |

---

## Project Structure

```
ScriptOCR/
├── ocr.py            # Phase 1 — CLI OCR engine
├── api.py            # Phase 2, 3 & 7c — FastAPI backend, shared pipeline, batch upload
├── field_mapper.py   # Phase 3 — Document field extraction logic (reads templates.py)
├── templates.json    # Phase 7 — Document schemas + export config (data, not code)
├── templates.py       # Phase 7 — Template store loader
├── exporters.py       # Phase 7 — Pluggable export destination layer (Google Sheets today)
├── sheets.py          # Phase 4 — Deprecated compatibility shim over exporters.py
├── index.html         # Phase 5 & 7c — Web UI (single-file + batch upload)
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- GitHub account (for free GPT-4o API access via GitHub Models)
- Google account (for Sheets integration)

### Installation

```
# Clone the repository
git clone https://github.com/Qasim-Bukhari/ScriptOCR.git
cd ScriptOCR

# Install dependencies
pip install fastapi uvicorn python-multipart opencv-python pillow openai aiofiles gspread google-auth
```

### Configuration

Set your GitHub token as an environment variable (not hardcoded in any file):

```
# Windows PowerShell
$env:GITHUB_TOKEN="your_github_token_here"

# Mac/Linux
export GITHUB_TOKEN="your_github_token_here"
```

To get a free GitHub token with GPT-4o access:

1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
2. Generate a new token (classic) with default scopes
3. Use it to access [GitHub Models](https://github.com/marketplace/models)

For Google Sheets integration, place a Google Service Account key as `credentials.json` in the project root (not included in this repo — see `.gitignore`).

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
| POST | `/ocr/fields/batch` | Same pipeline for up to 15 images at once (one document type per batch) |

### Example Request (single document)

```
curl -X POST http://127.0.0.1:8000/ocr/fields \
  -F "file=@document.jpg" \
  -F "document_type=store_request"
```

### Example Response

```
{
  "success": true,
  "filename": "document.jpg",
  "document_type": "store_request",
  "document_name": "Store Request Form",
  "extracted_text": "Store Request Form\nDate: 01 July 2026\n...",
  "fields": {
    "date": "01 July 2026",
    "requested_by": "Qasim Bukhari",
    "department": "Science Room",
    "item_name": "Whiteboard Markers",
    "quantity": "10 pieces",
    "purpose": "Required for daily classroom teaching",
    "signature": "Qasim"
  },
  "low_confidence_words": [],
  "confidence": "high",
  "sheet": {
    "success": false,
    "error": "No export destination configured for this document type"
  }
}
```

### Example Request (batch)

```
curl -X POST http://127.0.0.1:8000/ocr/fields/batch \
  -F "files=@doc1.jpg" \
  -F "files=@doc2.jpg" \
  -F "document_type=student_registration"
```

Returns a `summary` (`total`/`succeeded`/`failed`) plus a `results` array with one entry per image — a failure on one document never blocks the rest of the batch.

---

## Supported Document Types

| Key | Document |
|---|---|
| `store_request` | Store Request Form |
| `student_registration` | Student Registration Form |
| `procurement_request` | Procurement / Inventory Request |

New document types are added by editing `templates.json` — no Python changes required. `student_registration` also has a Google Sheets export configured; the others can be wired up the same way.

---

## Roadmap

- [x] Phase 1 — CLI OCR Engine
- [x] Phase 2 — FastAPI Backend
- [x] Phase 3 — Field Mapper
- [x] Phase 4 — Google Sheets Integration
- [x] Phase 5 — Web UI
- [x] Phase 7 — Data-Driven Templates & Pluggable Exporters
- [x] Phase 7c — Multi-Image Batch Upload
- [ ] Auto document type detection
- [ ] More NJV department schemas (real data collection)
- [ ] Multi-page single-document merging
- [ ] Server deployment
- [ ] Self-hosted fine-tuned OCR model

---

## Use Case

Inspired by real document workflows at **NJV High School, Karachi**, where multiple departments handle handwritten forms manually entered into external software. ScriptOCR automates that process — from image capture to structured data landing in a spreadsheet.

---

## Author

**Qasim Bukhari**
CS Graduate & Software Engineer | STEAM Educator, Karachi
