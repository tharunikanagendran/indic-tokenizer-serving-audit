# AI Usage Summary

**Submission:** AI Team Intern Assignment — The Audit  
**Author:** AI Team Audit Taskforce  

---

### Overview

In accordance with the ground rules, AI assistants (Gemini / Claude / Copilot) were utilized throughout this audit. This document provides an honest, transparent summary of where AI accelerated our analysis and where AI generated incorrect or misleading outputs that required manual verification and correction.

---

### Where AI Helped

1. **Boilerplate Code & Pipeline Automation**:
   - AI generated the initial structure for downloading and merging 4-way parallel sentences from AI4Bharat Samanantar (`prep_corpus.py`).
   - AI helped write the multi-tokenizer loop (`corrected_analysis.py`) to systematically compare `gpt2`, `cl100k_base`, `xlm-roberta-base`, `IndicBERTv2`, and `Qwen2.5-7B`.

2. **Mathematical Formulation & Table Formatting**:
   - AI assisted in converting KV cache arithmetic into clean GitHub-flavored markdown and KaTeX equations for `capacity_reconciliation.md`.
   - AI helped structure the decision framework for Part C into clear, explicitly labelled sections (Assumptions, Arithmetic, Success Metrics, Kill Criteria).

3. **Unicode Grapheme Pattern Recognition**:
   - AI correctly identified the Unicode Extended Grapheme Cluster definition (`\X`) for script-agnostic visual character counting.

---

### Where AI Misled & How It Was Caught

1. **Standard `re` Module Pattern Error (`bad escape \X`)**:
   - *What AI did*: AI generated `re.findall(r'\X', text)` using Python's built-in `re` module for grapheme cluster counting.
   - *How it broke*: Python's standard `re` module does not support the `\X` Unicode escape sequence, throwing `re.PatternError: bad escape \X`.
   - *Correction*: We caught the runtime stack trace, installed the third-party `regex` package, and replaced `import re` with `import regex`.

2. **Attempting to Access Gated HuggingFace Repositories**:
   - *What AI did*: AI suggested using `google/gemma-2b` and `ai4bharat/indic-bert` as open tokenizers for Part A3.
   - *How it broke*: Executing `from_pretrained` failed with `401 Client Error: GatedRepoError` because both repos require gated user agreement tokens.
   - *Correction*: We audited open HuggingFace repositories and identified non-gated, accessible multilingual/Indic tokenizers: `xlm-roberta-base`, `ai4bharat/IndicBERTv2-MLM-only`, and `Qwen/Qwen2.5-7B`.

3. **Mixing Decimal (GB) and Binary (GiB) Units in Memory Calculations**:
   - *What AI did*: AI's initial draft for Part B1 mixed decimal $10^9$ bytes ($24 \text{ GB}$) with binary $2^{30}$ bytes ($24 \text{ GiB}$), leading to ambiguous sequence capacity estimates (~25 vs ~29).
   - *Correction*: We re-derived the arithmetic from scratch, explicitly separating decimal metric calculation ($25 \text{ sequences}$) and binary GiB calculation ($29 \text{ sequences}$), demonstrating how both bounds align with preemption onset in `bench_log.csv` (between batch 24 and batch 32).
