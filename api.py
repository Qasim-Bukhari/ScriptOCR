"""
Handwriting OCR — Phase 3 (Phase 7 template store + exporter layer)
+ Phase (multi-image batch upload)
FastAPI Backend with Field Mapper integrated

Usage: uvicorn api:app --reload
"""

import os
import json
import base64
import tempfile
import asyncio
import time
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from openai import OpenAI

from field_mapper import map_fields
from templates import list_templates, get_template, TemplateNotFoundError
from exporters import get_exporter

# ── Configuration ─────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_ENDPOINT = "https://models.inference.ai.azure.com"
MODEL = "gpt-4o"

# Batch upload: bounded concurrency so a batch doesn't burst past
# GitHub Models' rate limit (Known Issue — GitHub Models Rate Limiting).
# GitHub Models' free tier enforces UserConcurrentRequests = 2 — confirmed
# in production via a 429 "RateLimitReached" error at concurrency 3.
BATCH_CONCURRENCY_LIMIT = 2
BATCH_MAX_FILES = 15

# Retry/backoff for transient rate-limit errors from GitHub Models. A single
# in-flight OCR/field-mapping call can still get a 429 even at concurrency 2
# (e.g. if you're also testing manually through the single-file endpoint at
# the same time), so this recovers automatically instead of failing that
# document outright.
RATE_LIMIT_MAX_RETRIES = 4
RATE_LIMIT_BASE_DELAY_SECONDS = 3

ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/jpg", "image/bmp", "image/tiff", "image/pjpeg"]
ALLOWED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".tiff", ".tif"]


def is_allowed_image(filename: str, content_type: str) -> bool:
    """True if the file looks like a supported image, checked by BOTH
    content_type and file extension.

    Needed because some formats — .jfif in particular — report inconsistent
    or missing content_type across browsers (e.g. 'image/pjpeg',
    'application/octet-stream', or empty), even though the actual bytes are
    ordinary JPEG data that OpenCV/GPT-4o handle fine. Falling back to the
    extension avoids rejecting valid images just because the browser's
    content_type guess was wrong.
    """
    if content_type in ALLOWED_IMAGE_TYPES:
        return True
    ext = os.path.splitext(filename or "")[-1].lower()
    return ext in ALLOWED_IMAGE_EXTENSIONS

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
    version="3.2.0"
)

_batch_semaphore = asyncio.Semaphore(BATCH_CONCURRENCY_LIMIT)


# ── Image Preprocessing ────────────────────────────────────────────────────────
def preprocess_image(image_path: str) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    binary = cv2.adaptiveThreshold(
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


# ── Rate-Limit Retry ────────────────────────────────────────────────────────
def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e)
    return "RateLimitReached" in msg or "429" in msg or "rate limit" in msg.lower()


def call_with_retry(func, *args, max_retries=RATE_LIMIT_MAX_RETRIES,
                     base_delay=RATE_LIMIT_BASE_DELAY_SECONDS, **kwargs):
    """Calls func(*args, **kwargs). If it fails with what looks like a
    transient rate-limit error (GitHub Models' 429 RateLimitReached), retries
    with exponential backoff (base_delay, base_delay*2, base_delay*4, ...).
    Any other exception is re-raised immediately — this is specifically for
    rate limits, not a general-purpose retry-everything wrapper.

    Runs inside a worker thread (via asyncio.to_thread), so the blocking
    time.sleep() here does not block the event loop or other concurrent
    documents.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if not _is_rate_limit_error(e) or attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
            last_error = e
    raise last_error


# ── Shared Pipeline (used by both single-file and batch endpoints) ────────────
async def process_single_document(contents: bytes, filename: str, document_type: str) -> dict:
    """Runs the full OCR -> field mapping -> export pipeline for one image.

    Never raises — always returns a dict with a 'success' key, so one bad
    image in a batch can't take down the rest of the batch. Blocking calls
    (cv2 preprocessing, GPT-4o requests, Sheets writes) are pushed to a
    thread via asyncio.to_thread so multiple documents can genuinely run
    concurrently instead of queueing behind each other on the event loop.
    """
    suffix = os.path.splitext(filename)[-1] or ".png"
    tmp_path = None
    preprocessed_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        preprocessed_path = await asyncio.to_thread(preprocess_image, tmp_path)
        ocr_result = await asyncio.to_thread(call_with_retry, extract_text, preprocessed_path)
        extracted_text = ocr_result.get("extracted_text", "")

        fields = await asyncio.to_thread(call_with_retry, map_fields, document_type, extracted_text)

        try:
            exporter = get_exporter(document_type)
            if exporter:
                sheet_result = await asyncio.to_thread(exporter.export, document_type, fields)
            else:
                sheet_result = {
                    "success": False,
                    "error": "No export destination configured for this document type"
                }
        except Exception as export_err:
            sheet_result = {"success": False, "error": str(export_err)}

        return {
            "success": True,
            "filename": filename,
            "extracted_text": extracted_text,
            "fields": fields,
            "low_confidence_words": ocr_result.get("low_confidence_words", []),
            "confidence": ocr_result.get("confidence", "unknown"),
            "sheet": sheet_result
        }
    except Exception as e:
        return {"success": False, "filename": filename, "error": str(e)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        if preprocessed_path and os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)


async def process_with_limit(contents: bytes, filename: str, document_type: str) -> dict:
    """Same as process_single_document, but bounded by BATCH_CONCURRENCY_LIMIT
    so a large batch can't fire all requests at GitHub Models at once."""
    async with _batch_semaphore:
        return await process_single_document(contents, filename, document_type)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/ui")
def ui():
    return FileResponse("index.html")


@app.get("/")
def root():
    return {
        "service": "Handwriting OCR API",
        "version": "3.2.0",
        "status": "running",
        "endpoints": {
            "POST /ocr": "Extract raw text from handwritten image",
            "POST /ocr/fields": "Extract and map fields from a single handwritten image",
            "POST /ocr/fields/batch": f"Extract and map fields from up to {BATCH_MAX_FILES} images at once",
            "GET /document-types": "List available document types",
            "GET /health": "Check service health"
        }
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/document-types")
def document_types():
    return {"document_types": list_templates()}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    if not is_allowed_image(file.filename, file.content_type):
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
    if not is_allowed_image(file.filename, file.content_type):
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")

    try:
        template = get_template(document_type)
    except TemplateNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    contents = await file.read()
    result = await process_single_document(contents, file.filename, document_type)

    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown processing error"))

    return JSONResponse(content={
        **result,
        "document_type": document_type,
        "document_name": template["name"],
    })


@app.post("/ocr/fields/batch")
async def ocr_fields_batch(
    files: List[UploadFile] = File(...),
    document_type: str = Form(...)
):
    """Processes a batch of images against the SAME document_type — e.g. a
    stack of student registration forms. Every image is treated as an
    independent document (its own row/result), not merged pages of one
    document. Concurrency is bounded by BATCH_CONCURRENCY_LIMIT; a failure
    on one image does not stop the rest of the batch."""
    try:
        template = get_template(document_type)
    except TemplateNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > BATCH_MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files: {len(files)}. Maximum per batch is {BATCH_MAX_FILES}."
        )

    tasks = []
    for f in files:
        if not is_allowed_image(f.filename, f.content_type):
            async def _invalid(filename=f.filename, content_type=f.content_type):
                return {
                    "success": False,
                    "filename": filename,
                    "error": f"Invalid file type: {content_type}"
                }
            tasks.append(_invalid())
            continue

        contents = await f.read()
        tasks.append(process_with_limit(contents, f.filename, document_type))

    results = await asyncio.gather(*tasks)

    succeeded = sum(1 for r in results if r["success"])
    failed = len(results) - succeeded

    return JSONResponse(content={
        "success": True,
        "document_type": document_type,
        "document_name": template["name"],
        "summary": {
            "total": len(results),
            "succeeded": succeeded,
            "failed": failed
        },
        "results": results
    })