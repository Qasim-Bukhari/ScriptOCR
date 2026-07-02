"""
Handwritten Document OCR — Phase 1
Stack: Python + OpenCV + GPT-4o-mini (via GitHub Models)
Usage: python ocr.py <image_path>
"""

import sys
import os
import json
import base64
import cv2
import numpy as np
from openai import OpenAI


# ── Configuration ─────────────────────────────────────────────────────────────

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_ENDPOINT = "https://models.inference.ai.azure.com"
GEMINI_MODEL = "gpt-4o" 

PROMPT = """You are an expert OCR system specialized in handwritten and cursive English text.

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

Return ONLY the JSON object."""


# ── Image Preprocessing (Optimized for Cursive Vision LLMs) ────────────────────

def preprocess_image(image_path: str) -> str:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # 1. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Bilateral Filter: Keep edges sharp, clean background noise
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # 3. Deskew alignment correction
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

    # 4. CLAHE Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # Slightly lowered clipLimit
    enhanced = clahe.apply(denoised)

    final_processed = cv2.adaptiveThreshold(
    enhanced, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
    )

    output_path = image_path.rsplit(".", 1)[0] + "_preprocessed.png"
    cv2.imwrite(output_path, final_processed)
    return output_path


# ── Text Extraction ───────────────────────────────────────────────────────────

def extract_text(image_path: str) -> dict:
    if not GITHUB_TOKEN:
        raise ValueError("Error: GITHUB_TOKEN environment variable is not set. Run: $env:GITHUB_TOKEN='your_token'")

    client = OpenAI(
        base_url=GITHUB_ENDPOINT,
        api_key=GITHUB_TOKEN
    )

    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model=GEMINI_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ]
    )

    raw = response.choices[0].message.content.strip()
    
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


# ── Display Result ─────────────────────────────────────────────────────────────

def display_result(result: dict):
    print("\n" + "═" * 60)
    print("  EXTRACTED TEXT")
    print("═" * 60)
    print(result.get("extracted_text", "No text extracted."))
    print("═" * 60)

    confidence = result.get("confidence", "unknown")
    low_conf   = result.get("low_confidence_words", [])

    print(f"\n  Confidence     : {confidence.upper()}")
    if low_conf:
        print(f"  Uncertain words: {', '.join(low_conf)}")
    else:
        print("  Uncertain words: None")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python ocr.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: File not found — {image_path}")
        sys.exit(1)

    print(f"\n  Processing: {image_path}")
    print("  [1/3] Preprocessing image (Grayscale + CLAHE)...")
    preprocessed_path = preprocess_image(image_path)

    print("  [2/3] Extracting text via GitHub Models...")
    try:
        result = extract_text(preprocessed_path)
        print("  [3/3] Done.\n")
        display_result(result)
    except Exception as e:
        print(f"\nAPI Error: {e}")
    finally:
        if os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)


if __name__ == "__main__":
    main()