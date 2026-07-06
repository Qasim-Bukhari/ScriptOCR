"""
ScriptOCR — LLM Provider Configuration

Single source of truth for which LLM provider/model api.py and
field_mapper.py talk to. Both OCR (image -> text) and field mapping
(text -> structured fields) go through whatever's configured here, so
switching providers means changing environment variables, not code.

Switch providers by setting LLM_PROVIDER in the environment:
  LLM_PROVIDER=github (default) — GitHub Models, needs GITHUB_TOKEN.
      50 requests/day PER MODEL, resets ~24h after first use that day.
  LLM_PROVIDER=gemini — Google Gemini free tier, needs GEMINI_API_KEY.
      Exposes an OpenAI-compatible endpoint, so the same OpenAI() client
      code works unchanged. ~1000-1500 requests/day on Flash models as of
      mid-2026 (verify current limits at ai.google.dev/gemini-api/docs/rate-limits
      since Google revises these without much notice).

Optionally override the model name itself with LLM_MODEL, independent of
provider (e.g. to try gpt-4o vs gpt-4o-mini on GitHub Models, or
gemini-2.5-flash vs gemini-2.0-flash on Gemini).
"""

import os

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "github").lower()

if LLM_PROVIDER == "gemini":
    LLM_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    LLM_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/"
    LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
else:
    LLM_API_KEY = os.environ.get("GITHUB_TOKEN", "")
    LLM_ENDPOINT = "https://models.inference.ai.azure.com"
    LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")