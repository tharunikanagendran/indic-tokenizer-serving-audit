# Chronological Engineering Lab Notebook

**Project:** AI Team Intern Assignment — The Audit  
**Author:** AI Team Audit Taskforce  
**Timeframe:** 5-Day Chronological Engineering Trajectory  

---

## Day 1: Environment Setup & Baseline Reproduction

### Goal
Setup working environment, inspect starter kit files, and reproduce the baseline numbers reported in `REPORT_v0.md`.

### Workspace Inspection
- Primary starter kit files: `fertility.py`, `REPORT_v0.md`, `bench/model_spec.md`, `bench/bench_log.csv`, `corpus_sample/`.
- Python dependencies installed: `tiktoken`, `transformers`, `sentencepiece`, `pyarrow`, `pandas`, `regex`.

### Initial Baseline Test
Ran `fertility.py` on the 10-sentence sample corpora using `gpt2`:
```bash
python starter_kit/fertility.py --corpus eng=starter_kit/corpus_sample/eng_sample.txt --corpus hin=starter_kit/corpus_sample/hin_sample.txt --tokenizer gpt2
```
- **Result**:
  - `eng`: 1.27 tok/word, 0.226 tok/char
  - `hin`: 7.45 tok/word, 1.579 tok/char
  - Ratio: `hin` is **5.89×** the fertility of `eng`.
  - Exactly matched `REPORT_v0.md`.

### Initial Hypotheses & Code Audit Ideas
- **Hypothesis 1.1**: Line 62 of `fertility.py` uses `line.split(" ")`. Standard text corpora contain multiple consecutive spaces or trailing spaces which will cause `split(" ")` to yield empty string `""` items in the list, inflating word count and artificially lowering fertility.
- **Hypothesis 1.2**: Macro-averaging (averaging ratios line by line) skews stats compared to micro-averaging (total tokens / total words across corpus).

---

## Day 2: Metric Audit, Bug Isolation & Counter-Evidence (Part A2)

### Experiment 2.1: Testing `line.split(" ")` vs `line.split()`
- **Command**: Executed `your-submission/partA/audit_script.py`.
- **Findings**:
  - On `eng_sample.txt`: `books  in` (double space on line 7).
  - Buggy `split(" ")`: English fertility = 1.2652 tok/word.
  - Fixed `split()`: English fertility = 1.2831 tok/word ($\Delta = +0.0179$, deflated word count by 1.4%).
  - Buggy `split(" ")`: Hindi fertility = 7.4485 tok/word.
  - Fixed `split()`: Hindi fertility = 7.5985 tok/word ($\Delta = +0.1500$, deflated word count by 2.0%).
- **Proof Sentence**: Multiple consecutive spaces create empty string elements in `split(" ")`, inflating word count and artificially deflating fertility by ~1.4–2.0%.

### Experiment 2.2: Macro-average vs Micro-average
- **Findings**:
  - English: Macro = 1.2831, Micro = 1.2692 tok/word ($\Delta = -0.0138$).
  - Hindi: Macro = 7.5985, Micro = 7.5246 tok/word ($\Delta = -0.0739$).
- **Proof Sentence**: Macro-averaging treats short and long sentences equally, allowing per-line ratio outliers to distort global token-to-word density by up to 3%.

### Experiment 2.3: `len(line)` Code Points vs Bytes vs Graphemes
- **Hypothesis**: `REPORT_v0` claim #2 ("tok/char agrees: 1.579 vs 0.226 = 7.0x worse per character, which confirms the per-word number") is conceptually false.
- **Findings**:
  - English: 1 code unit = 1 UTF-8 byte = 1 grapheme cluster (0.221 tok/unit).
  - Hindi: 1.583 tok/code_unit (7.16x ratio vs Eng), BUT 0.601 tok/byte (**2.72x ratio**) and 2.441 tok/grapheme (**11.05x ratio**).
- **Proof Sentence**: Python `len(line)` counts 1-byte ASCII code points for English but 3-byte code points for Hindi, misrepresenting 2.72x byte expansion as 7.16x character expansion and misleading leadership into thinking tok/char "confirmed" the tok/word ratio.

### Experiment 2.4: Impact of `.lower()` Transformation
- **Findings**:
  - English: Cased = 1.2308, Lowered = 1.2692 tok/word ($\Delta = +0.0385, +3.1\%$).
  - Hindi: Cased = 7.5246, Lowered = 7.5246 tok/word ($\Delta = 0.0000$).
- **Proof Sentence**: Lowercasing alters Latin subword splitting for capitalized words while having zero effect on casing-free Indic text, selectively inflating English tokenization.

### Experiment 2.5: Verifying Harmless Feature (NFC Normalization)
- **Tested**: `unicodedata.normalize("NFC", line)`.
- **Result**: Hindi raw tokens = 459, NFC normalized tokens = 459 ($\Delta = 0$).
- **Proof Sentence**: NFC normalization is completely harmless and necessary to standardize decomposed Unicode marks without introducing tokenization artifacts.

---

## Day 3: Corpus Construction & Multi-Tokenizer Evaluation (Part A1 & A3)

### Corpus Assembly (Part A1)
- Data Source: AI4Bharat Samanantar parallel corpora.
- Languages: English (`eng`), Hindi (`hin`), Kannada (`kan`), Tamil (`tam`).
- Script created: `your-submission/partA/prep_corpus.py`.
- Result: Assembled 1,000 clean 4-way parallel sentence tuples. Saved to `your-submission/partA/corpus/`.

### Multi-Tokenizer & Multi-Denominator Matrix (Part A3)
- Evaluated 5 tokenizers (`gpt2`, `cl100k_base`, `xlm-roberta-base`, `IndicBERTv2`, `Qwen2.5-7B`) across 4 denominators (`tok/word`, `tok/grapheme`, `tok/byte`, `tok/sentence`).
- Script created: `your-submission/partA/corrected_analysis.py`.
- Key Matrix Results (1,000 Parallel Sentences):

| Tokenizer | Metric | eng | hin | kan | tam | hin/eng ratio | kan/eng ratio | tam/eng ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **GPT-2** | `tok/sentence` | 10.03 | 67.75 | 131.29 | 163.88 | **6.75×** | **13.09×** | **16.34×** |
| | `tok/word` | 1.20 | 7.64 | 20.28 | 24.47 | **6.36×** | **16.88×** | **20.37×** |
| **IndicBERTv2** | `tok/sentence` | 9.60 | 10.54 | 9.65 | 10.17 | **1.10×** | **1.01×** | **1.06×** |
| | `tok/word` | 1.15 | 1.19 | 1.49 | 1.52 | **1.03×** | **1.30×** | **1.32×** |
| **XLM-RoBERTa** | `tok/sentence` | 10.69 | 12.15 | 13.22 | 14.61 | **1.14×** | **1.24×** | **1.37×** |
| | `tok/word` | 1.28 | 1.37 | 2.04 | 2.18 | **1.07×** | **1.60×** | **1.70×** |

### Key Insight on Denominator Selection
- **Winner**: **Tokens per Parallel Request (`tok/sentence`) under an Indic-aware tokenizer**.
- **Reasoning**: Dravidian languages (Kannada, Tamil) are agglutinative (combining prepositions and case markers into single words). Dividing by word count inflates Dravidian fertility because word count is ~22% smaller than English for identical meaning. `tok/sentence` holds semantic payload strictly constant. Under `IndicBERTv2`, Hindi is **1.10x** English and Tamil is **1.06x** English.

### Deliverable: Part A Memo
Written `your-submission/partA/memo.md` recommending a 1.15x–1.30x Indic cost allocation instead of 6.0x.

---

## Day 4: Capacity Reconciliation & Serving Log Analysis (Part B)

### Task B1: Theoretical KV Cache & Max Sequences Calculation
- **KV Cache per token**:
  $$\text{KV bytes/token} = 28 \text{ layers} \times 8 \text{ KV heads} \times 128 \text{ head\_dim} \times 2 \text{ (K+V)} \times 2 \text{ bytes (FP16)} = 114,688 \text{ bytes} = 112 \text{ KiB}$$
- **Max Sequences Calculation**:
  - $VRAM_{usable} = 24 \text{ GB} \times 0.92 = 22.08 \text{ GB}$
  - $M_{weights} = 4.2 \text{B} \times 2 = 8.40 \text{ GB}$
  - $M_{overhead} = 1.60 \text{ GB}$
  - $M_{KV\_pool} = 22.08 - 8.40 - 1.60 = 12.08 \text{ GB} = 12.08 \times 10^9 \text{ bytes}$
  - $KV_{seq} (4096 \text{ tokens}) = 4096 \times 114,688 = 469,762,048 \text{ bytes} = 0.46976 \text{ GB}$
  - $N_{max} = \lfloor 12.08 / 0.46976 \rfloor = \mathbf{25 \text{ sequences}}$ (or $29 \text{ sequences}$ in binary GiB).
- **Log Verification**: Log shows batch 24 has `kv_cache_util = 0.93` with 0 preemptions. Batch 32 triggers 7 preemptions (`kv_cache_util = 0.97`). Theoretical threshold matches empirical onset of preemption!

### Task B2: Long-Context Anomaly (Prompt 3584)
- **Observation**: Throughput peaks at batch 24 (1607.4 tok/s), then drops to 1384.0 tok/s at batch 32 and 1298.5 tok/s at batch 48.
- **Cause**: At batch 32+, KV cache pool is exhausted, forcing sequence preemption and prompt prefill re-execution (3,584 tokens recomputed per evicted request).
- **Recommendation**: Set `max_num_seqs = 24`. Eliminates preemption, restores throughput to 1607.4 tok/s, reduces e2e p95 latency by 34.4%.

### Task B3: `REPORT_v0` Misreading & Goodput Derivation
- **Misreading**: `reported_tok_s` included 87.5% prompt prefill tokens (3584 prompt / 4096 total).
- **Derivation of Goodput (Batch 24, Prompt 3584)**:
  - *Method 1 (Decode Wall Clock)*: $12,288 \text{ gen tokens} / (61.16\text{s} - 0.5005\text{s}) = \mathbf{202.57 \text{ gen tok/s}}$.
  - *Method 2 (ITL Inverse)*: $24 / 0.09607\text{s} = \mathbf{249.82 \text{ gen tok/s}}$.

### Task B4: Telemetry Counter
- Primary metric: `vllm:num_preemptions_total` (expected > 0 for batch >= 32).

Written `your-submission/partB/capacity_reconciliation.md`.

---

## Day 5: Decision Memo & Submission Finalization (Part C & Package)

### Decision Memo Strategy (Part C)
- Scenario: Casual/conversational tone across 6 Indic languages (Hindi, Kannada, Tamil, Telugu, Bengali, Marathi).
- Constraints: 1x A100-80GB (2 wks), 1 reviewer (Hindi+Kannada ONLY, 10h/wk), 3-wk launch, $0 API budget.
- Evaluated Paths:
  - Path (a) SFT: Fine-tuning risk of catastrophic forgetting & alignment drift on 4 un-reviewed languages.
  - Path (b) 1B Rewriter: Rejected due to +1.8s decode latency tax and double KV cache memory overhead.
  - Path (c) Few-Shot Prompt Engineering: Selected as primary path. Zero training cost, zero extra decode latency, instant iteration.
- Reviewer Math: $30 \text{ total hours} \times 15 \text{ resp/hr} = 450 \text{ evaluated responses}$.
- Success Metric: $\ge 75\%$ Casualness Win-Rate on Hindi/Kannada blind pairwise test.
- Kill Criterion: If by Day 7 casual win-rate $< 60\%$ or hallucination $> 5\%$, pivot to Path (a) QLoRA SFT on A100.
- Day 1 Experiment: 20-prompt test set comparing baseline vs 3 System Prompt personas on local A100.

Written `your-submission/partC/memo.md`.

---

## Final Verification & Self-Audit
- Checked all files against ground rules and evidence rule.
- Confirmed all required deliverables are present and formatted properly in `your-submission/`.
