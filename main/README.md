# LLM Tokenizer & Serving Capacity Audit

**Submission Repo for AI Team Intern Assignment — The Audit**  
**Author:** AI Team Audit Taskforce  
**Status:** Audit Complete & Defense Ready  

---

## Executive Overview

This repository contains a comprehensive technical audit of `REPORT_v0.md`, `fertility.py`, and `bench_log.csv`. 

Leadership was preparing to make routing and capacity decisions based on initial findings claiming that Hindi LLM serving costs **5.89× to 7.0× more than English**. Our empirical audit proves that this claim is an artifact of evaluating an outdated English-centric BPE tokenizer (`gpt2`) combined with flawed denominator choices (`tok/word` and Python code-unit `tok/char`).

On a 1,000-sentence parallel evaluation corpus covering English (`eng`), Hindi (`hin`), Kannada (`kan`), and Tamil (`tam`), **Hindi request token volume is only 1.10× to 1.14× English**, and Dravidian languages range from **1.01× to 1.37× English** under modern multilingual/Indic-aware tokenizers (`IndicBERTv2`, `XLM-RoBERTa`).

Furthermore, we reconciled serving load-test capacity logs (`bench_log.csv`), calculated exact KV-cache memory bounds ($112 \text{ KiB/token}$), identified the preemption thrashing threshold post batch 24, exposed prefill token misreadings in reported throughput, and designed an inference strategy decision framework for casual Indic model personas.

---

## Key Results Summary

### Part A: Tokenizer Audit & Denominator Correction
* Evaluated on 1,000 parallel sentences across English, Hindi, Kannada, and Tamil:

| Tokenizer | Metric | eng | hin | kan | tam | hin/eng ratio | kan/eng ratio | tam/eng ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GPT-2** (v0 baseline) | `tok/sentence` | 10.03 | 67.75 | 131.29 | 163.88 | **6.75×** | **13.09×** | **16.34×** |
| | `tok/word` | 1.20 | 7.64 | 20.28 | 24.47 | **6.36×** | **16.88×** | **20.37×** |
| **IndicBERTv2** (Indic-aware) | `tok/sentence` | 9.60 | 10.54 | 9.65 | 10.17 | **1.10×** | **1.01×** | **1.06×** |
| | `tok/word` | 1.15 | 1.19 | 1.49 | 1.52 | **1.03×** | **1.30×** | **1.32×** |
| **XLM-RoBERTa** (Multilingual) | `tok/sentence` | 10.69 | 12.15 | 13.22 | 14.61 | **1.14×** | **1.24×** | **1.37×** |
| | `tok/word` | 1.28 | 1.37 | 2.04 | 2.18 | **1.07×** | **1.60×** | **1.70×** |

* **Winning Metric**: **Tokens per Parallel Request (`tok/sentence`) under an Indic-aware Tokenizer**.  
  *Why*: `tok/sentence` holds semantic payload strictly constant across languages. `tok/word` penalizes agglutinative Dravidian languages (Kannada/Tamil pack 3–4 English words into 1 complex word).

### Part B: Capacity Reconciliation & Load Test Audit
* **KV-Cache Size**: $28 \text{ layers} \times 8 \text{ KV heads} \times 128 \text{ head\_dim} \times 2 \text{ (K+V)} \times 2 \text{ bytes (FP16)} = \mathbf{112 \text{ KiB/token}}$.
* **Max Concurrent 4096-Token Sequences**: $12.08 \text{ GB KV pool} / 0.46976 \text{ GB/seq} = \mathbf{25 \text{ sequences}}$ on 24GB L4 GPU.
* **Throughput Anomaly**: Preemption thrashing onset occurs at batch 32 (`kv_cache_util = 0.97`, 7 preempted seqs) due to KV memory exhaustion. Fix: set `max_num_seqs = 24`.
* **Throughput Misreading**: `reported_tok_s` included 87.5% prompt prefill tokens. True output generation goodput for batch 24 (prompt 3584) is **~202.57 gen tok/s** (decode wall clock) and **~249.82 gen tok/s** (ITL inverse).

### Part C: Casual Indic Persona Strategy
* **Recommendation**: **Path (c) Few-Shot Prompt Engineering** with a Day-7 Kill Criterion (casual win-rate $<60\% \implies$ pivot to QLoRA SFT on A100).
* **Constraints**: 1× A100-80GB (2 wks), 1 native reviewer (Hindi+Kannada, 10h/wk = 450 total evaluated responses), $0 API budget.

---

## Repository Structure

```
your-submission/
├── README.md                          # Main project overview & quickstart
├── NOTEBOOK.md                        # Chronological 5-day engineering log (graded)
├── AI_USAGE.md                        # Honest summary of AI help and hallucinations caught
├── partA/                             # Tokenizer Audit Code & Artifacts
│   ├── prep_corpus.py                 # Script to build 1000 4-way parallel sentence corpus
│   ├── audit_script.py                # Script isolating fertility.py code bugs with exact deltas
│   ├── corrected_analysis.py          # Script evaluating 5 tokenizers across 4 denominators
│   ├── corrected_metrics_matrix.csv   # Complete empirical evaluation matrix CSV
│   ├── memo.md                        # Executive Recommendation Memo (Part A4, <= 1 page)
│   └── corpus/                        # Clean parallel eval corpora (eng, hin, kan, tam)
├── partB/                             # Serving Capacity Reconciliation
│   └── capacity_reconciliation.md     # Detailed KV math, log derivations, & telemetry metrics
└── partC/                             # Product & Inference Strategy
    └── memo.md                        # Casual Indic Persona Decision Memo (<= 1 page)
```

---

## Environment Setup & Quickstart

### 1. Requirements & Dependencies
Ensure Python 3.10+ is installed along with required packages:

```bash
pip install tiktoken transformers sentencepiece pyarrow pandas regex huggingface_hub
```

### 2. Reproduce Tokenizer Audit & Evidence Demos (Part A2)
To run the script that isolates `fertility.py` code bugs and measures exact before/after deltas:

```bash
python your-submission/partA/audit_script.py
```

### 3. Build Parallel Eval Corpus (Part A1)
To re-generate the 1,000-sentence 4-way parallel corpus from AI4Bharat Samanantar:

```bash
python your-submission/partA/prep_corpus.py
```

### 4. Run Multi-Tokenizer Evaluation Matrix (Part A3)
To compute the full matrix of tokenizers ($\times$) denominators across 1,000 parallel sentences:

```bash
python your-submission/partA/corrected_analysis.py
```

---

## Primary File References

- **Lab Log**: [`NOTEBOOK.md`](NOTEBOOK.md)
- **AI Tooling Transparency**: [`AI_USAGE.md`](AI_USAGE.md)
- **Tokenizer Memo**: [`partA/memo.md`](partA/memo.md)
- **Capacity Report**: [`partB/capacity_reconciliation.md`](partB/capacity_reconciliation.md)
- **Decision Memo**: [`partC/memo.md`](partC/memo.md)
