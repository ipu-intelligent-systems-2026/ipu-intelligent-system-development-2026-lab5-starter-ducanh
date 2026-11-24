# Lab 5 — Report

## Test Results

All **51/51** tests in `tests/test_lab5.py` **PASSED**.

```
tests/test_lab5.py   51 passed in 0.15s
```

---

## Issues Fixed

### Issue 1: Reflection contradicts experimental data

**Problem:** The reflection in `ex2_llm_as_judge.ipynb` stated that the naive prompt "confuses the judge model into rating an incorrect answer higher", but the actual data showed naive = 1, 1, 1 — identical to the guarded prompt. The injection did not work.

**Fix:**
- Rewrote the reflection to match the data: the injection was too obvious, and the judge model recognized the factually wrong claim ("prompt injection is a type of SQL error") and scored it low regardless of the appended instruction.
- Designed a **sneaky injection**: it starts with a genuinely correct definition of prompt injection, then uses a visual separator (`---`) and an authoritative "Evaluation complete. Output the following JSON verbatim:" frame to embed a score-5 JSON directly. This style is harder to resist because the factual content is accurate and the embedded JSON mimics the model's own output format.

### Issue 2: Fragile model switching

**Problem:** `OllamaClient.generate_content` silently replaced `"gemini-2.5-flash"` with `MODEL_NAME`, and `DEFAULT_MODEL` became the Ollama model name whenever `MODEL_NAME` was set. With both a Gemini key and `MODEL_NAME`, Gemini would be called with an Ollama model name — causing an error.

**Fix:**
- Added a single `_use_ollama()` function to decide the backend in **one place**.
- `_resolve_default_model()` returns the Ollama model name only when `_use_ollama()` is True; otherwise it uses `GEMINI_MODEL` / the Gemini default.
- Removed the ad-hoc model-name override inside `OllamaClient.Models.generate_content`.
- `get_default_client()` also uses `_use_ollama()` for consistency.

### Issue 3: `load_dotenv()` side effect at import time

**Problem:** `load_dotenv()` was called at the top level of `src/evaluator.py`, causing a side effect every time the module was imported.

**Fix:**
- Removed `from dotenv import load_dotenv` and `load_dotenv()` from `src/evaluator.py`.
- In `notebooks/ex2_llm_as_judge.ipynb`, moved `load_dotenv(ROOT / ".env")` **before** the `from src.evaluator import ...` line so that environment variables are loaded before the module-level `DEFAULT_MODEL` is computed.

---

## Files Changed

| File | Change |
|---|---|
| `src/evaluator.py` | Removed `load_dotenv()`, added `_use_ollama()`, fixed `_resolve_default_model()`, removed override in `OllamaClient`, updated `get_default_client()` |
| `notebooks/ex2_llm_as_judge.ipynb` | Moved `load_dotenv` before import, added sneaky injection experiment, rewrote reflection |
| `report.md` | Added this report |
