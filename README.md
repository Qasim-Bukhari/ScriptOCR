# ScriptOCR

A document OCR and field extraction system that converts handwritten and printed documents into structured data. Built as a portfolio project inspired by real document workflows at NJV High School, Karachi.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red) ![GPT-4o](https://img.shields.io/badge/GPT--4o-Vision-orange)

---

## What it does

Upload an image of any handwritten or printed document. ScriptOCR preprocesses the image, extracts all text using GPT-4o vision, and intelligently maps the extracted text to the specific fields of the document type — returning clean, structured JSON data ready for further processing or software integration.

---

## Features

- **Image Preprocessing** — Grayscale conversion, denoising, deskewing, and contrast enhancement via OpenCV
- **Handwriting OCR** — Powered by GPT-4o vision for high accuracy on cursive and printed text
- **Field Extraction** — Intelligent field mapping per document type using structured prompts
- **REST API** — FastAPI backend with endpoints for OCR and field extraction
- **Web UI** — Clean, professional side-by-side interface for document upload and results
- **Multiple Document Types** — Store requests, student registration forms, procurement requests
- **Confidence Scoring** — Each extraction returns a confidence level and flags uncertain words

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Image Processing | OpenCV, Pillow |
| OCR Engine | GPT-4o via GitHub Models |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |

---

## Project Structure

```
ScriptOCR/
├── ocr.py            # Phase 1 — CLI OCR engine
├── api.py            # Phase 2 & 3 — FastAPI backend with field mapper
├── field_mapper.py   # Phase 3 — Document field extraction logic
├── index.html        # Phase 5 — Web UI
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- GitHub account (for free GPT-4o API access via GitHub Models)

### Installation

```bash
# Clone the repository
git clone https://github.com/Qasim-Bukhari/ScriptOCR.git
cd ScriptOCR

# Install dependencies
pip install fastapi uvicorn python-multipart opencv-python pillow openai aiofiles
```

### Configuration

Open `api.py` and `field_mapper.py` and set your GitHub token:

```python
GITHUB_TOKEN = "your_github_token_here"
```

To get a free GitHub token with GPT-4o access:
1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
2. Generate a new token (classic) with default scopes
3. Use it to access [GitHub Models](https://github.com/marketplace/models)

### Run the server

```bash
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
| POST | `/ocr` | Extract raw text from image |
| POST | `/ocr/fields` | Extract and map fields from image |

### Example Request

```bash
curl -X POST http://127.0.0.1:8000/ocr/fields \
  -F "file=@document.jpg" \
  -F "document_type=store_request"
```

### Example Response

```json
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
  "confidence": "high"
}
```

---

## Supported Document Types

| Key | Document |
|---|---|
| `store_request` | Store Request Form |
| `student_registration` | Student Registration Form |
| `procurement_request` | Procurement / Inventory Request |

New document types can be added by defining a schema in `field_mapper.py`.

---

## Roadmap

- [x] Phase 1 — CLI OCR Engine
- [x] Phase 2 — FastAPI Backend
- [x] Phase 3 — Field Mapper
- [ ] Phase 4 — Integration Layer
- [x] Phase 5 — Web UI
- [ ] Auto document type detection
- [ ] Self-hosted fine-tuned OCR model
- [ ] Processing history and logs

---

## Use Case

Inspired by real document workflows at **NJV High School, Karachi**, where multiple departments handle handwritten forms manually entered into external software. ScriptOCR automates that process — from image capture to structured data extraction.

---

## Author

**Qasim Bukhari**
CS Graduate & Software Engineer | STEAM Educator, Karachi
