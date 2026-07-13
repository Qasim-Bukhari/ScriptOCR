"""
Handwriting OCR — Phase 3 (Phase 7 template store + exporter layer)
+ Phase (multi-image batch upload)
FastAPI Backend with Field Mapper integrated

Usage: uvicorn api:app --reload
"""

from dotenv import load_dotenv
load_dotenv()

import os
import json
import base64
import tempfile
import asyncio
import time
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from field_mapper import map_fields, extract_and_map_fields
from templates import list_templates, get_template, TemplateNotFoundError
from exporters import get_exporter

# ── Configuration ─────────────────────────────────────────────────────────────
# ── LLM Provider Configuration ────────────────────────────────────────────────
# Shared with field_mapper.py via llm_config.py — see that file for how to
# switch providers (e.g. when a daily free-tier quota runs out).
from llm_config import LLM_MODEL as MODEL, llm_client

# Batch upload: bounded concurrency so a batch doesn't burst past
# GitHub Models' rate limit (Known Issue — GitHub Models Rate Limiting).
# GitHub Models' free tier enforces UserConcurrentRequests = 2 — confirmed
# in production via a 429 "RateLimitReached" error at concurrency 3.
BATCH_CONCURRENCY_LIMIT = int(os.environ.get("BATCH_CONCURRENCY_LIMIT", "2"))
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

# ── Authentication ────────────────────────────────────────────────────────────
import secrets
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()
APP_USERNAME = os.environ.get("APP_USERNAME", "njv")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")

# Fail loudly at startup if APP_PASSWORD is unset, instead of silently
# allowing a blank password to authenticate successfully. Without this
# check, `secrets.compare_digest(credentials.password, "")` would succeed
# for anyone who submits an empty password field — a real risk if this
# env var is ever forgotten during a fresh deploy (e.g. a new host).
if not APP_PASSWORD:
    raise RuntimeError(
        "APP_PASSWORD environment variable is not set. Refusing to start "
        "with authentication effectively disabled. Set APP_PASSWORD in "
        "your .env file (local) or your host's environment variables "
        "(deployed) before starting the server."
    )

def require_auth(credentials: HTTPBasicCredentials = Depends(security)):
    correct_user = secrets.compare_digest(credentials.username, APP_USERNAME)
    correct_pass = secrets.compare_digest(credentials.password, APP_PASSWORD)
    if not (correct_user and correct_pass):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"}
        )
    return credentials.username

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Handwriting OCR API",
    description="Extract and map fields from handwritten document images.",
    version="3.2.0"
)

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_batch_semaphore = asyncio.Semaphore(BATCH_CONCURRENCY_LIMIT)
_export_lock = asyncio.Lock()


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
    client = llm_client

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


# ── Rate-Limit + Transient-Network Retry ────────────────────────────────────
def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e)
    return "RateLimitReached" in msg or "429" in msg or "rate limit" in msg.lower()


def _is_transient_network_error(e: Exception) -> bool:
    """Catches timeouts and connection failures — e.g. a flaky local network
    or DNS resolver dropping a request mid-flight. These are just as
    transient as a 429 and worth retrying, unlike a real API/schema error."""
    msg = str(e).lower()
    name = type(e).__name__
    return (
        "timeout" in msg or "timed out" in msg
        or "connection" in msg or "connect" in msg
        or "nameresolutionerror" in msg or "getaddrinfo" in msg
        or name in ("APITimeoutError", "APIConnectionError", "ConnectionError", "Timeout")
    )


def call_with_retry(func, *args, max_retries=RATE_LIMIT_MAX_RETRIES,
                     base_delay=RATE_LIMIT_BASE_DELAY_SECONDS, **kwargs):
    """Calls func(*args, **kwargs). If it fails with what looks like a
    transient error — GitHub Models' 429 RateLimitReached, or a timeout/
    connection blip (flaky local network, DNS hiccup) — retries with
    exponential backoff (base_delay, base_delay*2, base_delay*4, ...).
    Any other exception (bad input, real API error) is re-raised
    immediately — this isn't a general-purpose retry-everything wrapper.

    Prints when a retry actually fires, with which function, which
    attempt, and how long it's sleeping — otherwise these backoff delays
    are invisible in the terminal and just look like unexplained slowness
    (this was found after a 5-document batch took ~140s when ~45-60s was
    expected, with no way to tell whether retries were the cause).

    Runs inside a worker thread (via asyncio.to_thread), so the blocking
    time.sleep() here does not block the event loop or other concurrent
    documents.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if not (_is_rate_limit_error(e) or _is_transient_network_error(e)) or attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"[retry] {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): "
                  f"{type(e).__name__}: {e} — retrying in {delay}s")
            time.sleep(delay)
            last_error = e
    raise last_error


# ── Shared preprocessing step (used by every pipeline) ─────────────────────
async def _preprocess(contents: bytes, filename: str) -> str:
    """Writes image bytes to a temp file, preprocesses it, and returns the
    preprocessed file's path. The raw temp file is cleaned up immediately;
    the caller owns cleanup of the returned preprocessed file."""
    suffix = os.path.splitext(filename)[-1] or ".png"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        return await asyncio.to_thread(preprocess_image, tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


async def _run_ocr(contents: bytes, filename: str) -> dict:
    """OCR-only (no field mapping) — extracted_text/low_confidence_words/
    confidence, nothing else. Used ONLY by merge mode's per-page step,
    where fields can't be mapped yet because a multi-page document's
    fields may span several images (see /ocr/fields/merge). Raises on
    failure — callers decide how to handle that.
    """
    preprocessed_path = await _preprocess(contents, filename)
    try:
        return await asyncio.to_thread(call_with_retry, extract_text, preprocessed_path)
    finally:
        if os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)


async def _run_extract_and_map(contents: bytes, filename: str, document_type: str) -> dict:
    """OCR + field mapping in ONE model call (extract_and_map_fields in
    field_mapper.py), instead of two sequential round-trips. Used by the
    single-document and batch pipelines, where one image = one document,
    so OCR and field mapping can safely happen together. Roughly halves
    per-document latency and API-quota usage vs. the old two-call flow.
    Raises on failure — callers decide how to handle that.
    """
    print(f"[timing] {filename}: raw upload size={len(contents) / 1024:.0f}KB")
    prep_start = time.monotonic()
    preprocessed_path = await _preprocess(contents, filename)
    prep_elapsed = time.monotonic() - prep_start
    preprocessed_size = os.path.getsize(preprocessed_path) / 1024
    print(f"[timing] {filename}: preprocessing={prep_elapsed:.1f}s, "
          f"preprocessed size={preprocessed_size:.0f}KB")
    try:
        model_start = time.monotonic()
        result = await asyncio.to_thread(call_with_retry, extract_and_map_fields, preprocessed_path, document_type)
        model_elapsed = time.monotonic() - model_start
        print(f"[timing] {filename}: model call={model_elapsed:.1f}s")
        return result
    finally:
        if os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)


async def ocr_only(contents: bytes, filename: str) -> dict:
    """Same OCR step as _run_ocr, but never raises — always returns a dict
    with a 'success' key. Used by merge mode, where each page is OCR'd
    independently before being combined (see /ocr/fields/merge)."""
    try:
        ocr_result = await _run_ocr(contents, filename)
        return {
            "success": True,
            "filename": filename,
            "extracted_text": ocr_result.get("extracted_text", ""),
            "low_confidence_words": ocr_result.get("low_confidence_words", []),
            "confidence": ocr_result.get("confidence", "unknown"),
        }
    except Exception as e:
        return {"success": False, "filename": filename, "error": str(e)}


async def ocr_only_with_limit(contents: bytes, filename: str) -> dict:
    """Same as ocr_only, but bounded by BATCH_CONCURRENCY_LIMIT — merge mode
    reuses the same semaphore as batch mode since it hits the same GitHub
    Models rate limit."""
    async with _batch_semaphore:
        return await ocr_only(contents, filename)


CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def worst_confidence(confidences: List[str]) -> str:
    """Returns the lowest confidence among a list of per-page confidences.
    Used by merge mode to roll up one overall confidence for the merged
    document — if any page came back 'low', the merged document is 'low'."""
    ranked = [c for c in confidences if c in CONFIDENCE_RANK]
    if not ranked:
        return "unknown"
    return min(ranked, key=lambda c: CONFIDENCE_RANK[c])


# ── Shared Pipeline (used by both single-file and batch endpoints) ────────────
async def process_single_document(contents: bytes, filename: str, document_type: str) -> dict:
    """Runs the full OCR+field-mapping -> export pipeline for one image.
    OCR and field mapping now happen in a single model call (see
    _run_extract_and_map) instead of two sequential ones — this was the
    single biggest latency/quota win available, since it cuts one full
    model round-trip per document.

    Never raises — always returns a dict with a 'success' key, so one bad
    image in a batch can't take down the rest of the batch.
    """
    try:
        start = time.monotonic()
        result = await _run_extract_and_map(contents, filename, document_type)
        extract_elapsed = time.monotonic() - start
        extracted_text = result.get("extracted_text", "")
        fields = result.get("fields", {})

        export_start = time.monotonic()
        try:
            exporter = get_exporter(document_type)
            if exporter:
                async with _export_lock:
                    sheet_result = await asyncio.to_thread(call_with_retry, exporter.export, document_type, fields)
            else:
                sheet_result = {
                    "success": False,
                    "error": "No export destination configured for this document type"
                }
        except Exception as export_err:
            sheet_result = {"success": False, "error": str(export_err)}
        export_elapsed = time.monotonic() - export_start

        print(f"[timing] {filename}: extract+map={extract_elapsed:.1f}s, export={export_elapsed:.1f}s, "
              f"total={extract_elapsed + export_elapsed:.1f}s")

        return {
            "success": True,
            "filename": filename,
            "extracted_text": extracted_text,
            "fields": fields,
            "low_confidence_words": result.get("low_confidence_words", []),
            "confidence": result.get("confidence", "unknown"),
            "sheet": sheet_result
        }
    except Exception as e:
        return {"success": False, "filename": filename, "error": str(e)}


async def process_with_limit(contents: bytes, filename: str, document_type: str) -> dict:
    """Same as process_single_document, but bounded by BATCH_CONCURRENCY_LIMIT
    so a large batch can't fire all requests at GitHub Models at once."""
    async with _batch_semaphore:
        return await process_single_document(contents, filename, document_type)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/ui", dependencies=[Depends(require_auth)])
def ui():
    return FileResponse("index.html")


@app.get("/", dependencies=[Depends(require_auth)])
def root():
    return {
        "service": "Handwriting OCR API",
        "version": "3.2.0",
        "status": "running",
        "endpoints": {
            "POST /ocr": "Extract raw text from handwritten image",
            "POST /ocr/fields": "Extract and map fields from a single handwritten image",
            "POST /ocr/fields/batch": f"Extract and map fields from up to {BATCH_MAX_FILES} images at once (independent documents)",
            "POST /ocr/fields/merge": f"Merge up to {BATCH_MAX_FILES} page images of ONE document into a single extraction",
            "GET /document-types": "List available document types",
            "GET /health": "Check service health"
        }
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/document-types", dependencies=[Depends(require_auth)])
def document_types():
    return {"document_types": list_templates()}


@app.post("/ocr", dependencies=[Depends(require_auth)])
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


@app.post("/ocr/fields", dependencies=[Depends(require_auth)])
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


@app.post("/ocr/fields/batch", dependencies=[Depends(require_auth)])
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

    batch_start = time.monotonic()
    results = await asyncio.gather(*tasks)
    batch_elapsed = time.monotonic() - batch_start
    print(f"[timing] batch of {len(files)} files: total wall-clock={batch_elapsed:.1f}s "
          f"(concurrency limit={BATCH_CONCURRENCY_LIMIT})")

    succeeded = sum(1 for r in results if r["success"])
    failed = len(results) - succeeded
    saved = sum(1 for r in results if r.get("sheet", {}).get("success"))

    return JSONResponse(content={
        "success": True,
        "document_type": document_type,
        "document_name": template["name"],
        "summary": {
            "total": len(results),
            "succeeded": succeeded,
            "failed": failed,
            "saved": saved
        },
        "results": results
    })


@app.post("/ocr/fields/merge", dependencies=[Depends(require_auth)])
async def ocr_fields_merge(
    files: List[UploadFile] = File(...),
    document_type: str = Form(...)
):
    """Merges multiple images that are PAGES OF ONE document (e.g. a 2-page
    form) into a single extraction + a single exported row — distinct from
    /ocr/fields/batch, which treats every image as an independent document.

    Each page is OCR'd independently (same preprocessing/retry as every
    other endpoint), in upload order. The extracted texts are then
    concatenated with page markers and run through field mapping ONCE,
    so a field split across pages (label on page 1, value on page 2) can
    still be matched correctly. Export also happens ONCE, producing a
    single row.

    Unlike batch mode, this does NOT isolate per-page failures: if any
    page fails OCR, the whole request fails with a 400/500 naming the
    page, rather than exporting a row with that page's fields silently
    missing — a partial multi-page form isn't a usable record.
    """
    try:
        template = get_template(document_type)
    except TemplateNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > BATCH_MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many pages: {len(files)}. Maximum per document is {BATCH_MAX_FILES}."
        )

    # OCR every page independently, bounded by the same concurrency limit
    # as batch mode (same underlying GitHub Models rate limit). Tasks are
    # built in upload order and asyncio.gather preserves that order in its
    # results, so page order survives the concurrency.
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
        tasks.append(ocr_only_with_limit(contents, f.filename))

    page_results = await asyncio.gather(*tasks)

    failed_pages = [p for p in page_results if not p["success"]]
    if failed_pages:
        first = failed_pages[0]
        raise HTTPException(
            status_code=500,
            detail=f"Page '{first['filename']}' failed OCR: {first.get('error', 'Unknown error')}"
        )

    merged_text = "\n\n".join(
        f"--- Page {i + 1} ({p['filename']}) ---\n{p['extracted_text']}"
        for i, p in enumerate(page_results)
    )

    all_low_confidence = []
    for p in page_results:
        all_low_confidence.extend(p.get("low_confidence_words", []))
    overall_confidence = worst_confidence([p.get("confidence", "unknown") for p in page_results])

    try:
        fields = await asyncio.to_thread(call_with_retry, map_fields, document_type, merged_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Field mapping failed: {e}")

    try:
        exporter = get_exporter(document_type)
        if exporter:
            async with _export_lock:
                sheet_result = await asyncio.to_thread(call_with_retry, exporter.export, document_type, fields)
        else:
            sheet_result = {
                "success": False,
                "error": "No export destination configured for this document type"
            }
    except Exception as export_err:
        sheet_result = {"success": False, "error": str(export_err)}

    return JSONResponse(content={
        "success": True,
        "document_type": document_type,
        "document_name": template["name"],
        "page_count": len(files),
        "pages": [
            {
                "filename": p["filename"],
                "confidence": p.get("confidence", "unknown"),
                "low_confidence_words": p.get("low_confidence_words", [])
            }
            for p in page_results
        ],
        "extracted_text": merged_text,
        "fields": fields,
        "low_confidence_words": all_low_confidence,
        "confidence": overall_confidence,
        "sheet": sheet_result
    })