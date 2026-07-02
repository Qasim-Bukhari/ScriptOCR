"""
Handwriting OCR — Phase 3
FastAPI Backend with Field Mapper integrated
Usage: uvicorn api:app --reload
"""

import os
import json
import base64
import tempfile
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from field_mapper import map_fields, DOCUMENT_SCHEMAS
from sheets import push_to_sheet


# ── Configuration ─────────────────────────────────────────────────────────────

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_ENDPOINT = "https://models.inference.ai.azure.com"
MODEL           = "gpt-4o"

OCR_PROMPT = """You are an expert OCR system specialized in handwritten and cursive English text.

Your task:
1. Extract ALL handwritten text from the image exactly as written.
2. Preserve the original line breaks, bullet points, and indentations.
3. Pay close attention to connected cursive script and symbols (like / or -).
4. For any word you are genuinely uncertain about, wrap it in [?word?] markers.
5. Do NOT add any commentary, explanation, or extra text.

Return your response as a JSON object in this exact format:
{
  "extracted_text": "the full extracted text here",
  "low_confidence_words": ["word1", "word2"],
  "confidence": "high | medium | low"
}

Return ONLY the JSON object. No markdown, no backticks, no extra text."""


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Handwriting OCR API",
    description="Extract and map fields from handwritten document images.",
    version="3.0.0"
)


# ── Image Preprocessing ────────────────────────────────────────────────────────

def preprocess_image(image_path: str) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image.")

    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    coords = np.column_stack(np.where(denoised < 200))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 30:
            (h, w) = denoised.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            denoised = cv2.warpAffine(denoised, M, (w, h),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE)

    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    binary   = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    output_path = image_path.rsplit(".", 1)[0] + "_preprocessed.png"
    cv2.imwrite(output_path, binary)
    return output_path


# ── OCR ───────────────────────────────────────────────────────────────────────

def extract_text(image_path: str) -> dict:
    client = OpenAI(base_url=GITHUB_ENDPOINT, api_key=GITHUB_TOKEN)

    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": OCR_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
        ]}]
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw.strip())


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/ui")
def ui():
    return FileResponse("index.html")


@app.get("/")
def root():
    return {
        "service": "Handwriting OCR API",
        "version": "3.0.0",
        "status": "running",
        "endpoints": {
            "POST /ocr": "Extract raw text from handwritten image",
            "POST /ocr/fields": "Extract and map fields from handwritten image",
            "GET /document-types": "List available document types",
            "GET /health": "Check service health"
        }
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/document-types")
def document_types():
    return {
        "document_types": {
            key: value["name"]
            for key, value in DOCUMENT_SCHEMAS.items()
        }
    }


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/bmp", "image/tiff"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")

    suffix = os.path.splitext(file.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    preprocessed_path = None
    try:
        preprocessed_path = preprocess_image(tmp_path)
        result = extract_text(preprocessed_path)
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "extracted_text": result.get("extracted_text", ""),
            "low_confidence_words": result.get("low_confidence_words", []),
            "confidence": result.get("confidence", "unknown")
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if preprocessed_path and os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)


@app.post("/ocr/fields")
async def ocr_fields(
    file: UploadFile = File(...),
    document_type: str = Form(...)
):
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/bmp", "image/tiff"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")

    if document_type not in DOCUMENT_SCHEMAS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown document type: '{document_type}'. "
                   f"Available: {list(DOCUMENT_SCHEMAS.keys())}"
        )

    suffix = os.path.splitext(file.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    preprocessed_path = None
    try:
        preprocessed_path = preprocess_image(tmp_path)
        ocr_result = extract_text(preprocessed_path)
        extracted_text = ocr_result.get("extracted_text", "")
        fields = map_fields(document_type, extracted_text)

        # Push to Google Sheets
        sheet_result = None
        try:
            sheet_result = push_to_sheet(document_type, fields)
        except Exception as sheet_err:
            sheet_result = {"success": False, "error": str(sheet_err)}

        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "document_type": document_type,
            "document_name": DOCUMENT_SCHEMAS[document_type]["name"],
            "extracted_text": extracted_text,
            "fields": fields,
            "low_confidence_words": ocr_result.get("low_confidence_words", []),
            "confidence": ocr_result.get("confidence", "unknown"),
            "sheet": sheet_result
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if preprocessed_path and os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)
