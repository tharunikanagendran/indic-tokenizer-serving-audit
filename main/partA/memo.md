# Executive Memo: Tokenizer Fertility Audit & Indic Routing Strategy

**To:** Leadership & Infrastructure Engineering  
**From:** AI Team Audit Taskforce  
**Date:** September 6, 2026  
**Subject:** Corrected Tokenizer Fertility Benchmark & Indic Traffic Routing Recommendation  

---

### Executive Summary

`REPORT_v0.md` concluded that Hindi serving costs are **5.89× to 7.0× higher** than English and recommended budgeting a 6× serving multiplier for all Indic traffic. Our audit demonstrates that this 6× multiplier is an artifact of evaluating a 2019 English-centric tokenizer (`gpt2`) combined with flawed denominator choices (`tok/word` and Python code-unit `tok/char`).

On a proper 1,000-sentence parallel eval corpus covering English, Hindi, Kannada, and Tamil evaluated on modern multilingual and Indic-aware tokenizers (`XLM-RoBERTa` / `IndicBERTv2`), **Hindi request token volume is only 1.10× to 1.14× English**, and Dravidian languages (Kannada, Tamil) range from **1.01× to 1.37× English**. 

Budgeting a 6× multiplier will result in severe over-provisioning and misallocated GPU resources.

---

### Corrected Headline Numbers

*Evaluated on 1,000 parallel sentences across English (eng), Hindi (hin), Kannada (kan), and Tamil (tam):*

| Tokenizer | Metric | eng | hin | kan | tam | hin/eng ratio | kan/eng ratio | tam/eng ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GPT-2** (v0 baseline) | `tok / sentence` | 10.03 | 67.75 | 131.29 | 163.88 | **6.75×** | **13.09×** | **16.34×** |
| | `tok / word` | 1.20 | 7.64 | 20.28 | 24.47 | **6.36×** | **16.88×** | **20.37×** |
| **IndicBERTv2** (Indic-aware) | `tok / sentence` | 9.60 | 10.54 | 9.65 | 10.17 | **1.10×** | **1.01×** | **1.06×** |
| | `tok / word` | 1.15 | 1.19 | 1.49 | 1.52 | **1.03×** | **1.30×** | **1.32×** |
| **XLM-RoBERTa** (Multilingual) | `tok / sentence` | 10.69 | 12.15 | 13.22 | 14.61 | **1.14×** | **1.24×** | **1.37×** |
| | `tok / word` | 1.28 | 1.37 | 2.04 | 2.18 | **1.07×** | **1.60×** | **1.70×** |

---

### Key Metric & Routing Recommendation

1. **Primary Metric for Routing & Capacity**: **Tokens per Parallel Request (`tok / sentence`) under a Multilingual/Indic-aware Vocabulary**.
   - *Why*: Whitespace word count (`tok/word`) penalizes morphologically agglutinative Dravidian languages (Kannada/Tamil pack 3–4 English words into 1 complex word, artificially inflating per-word fertility). `tok/sentence` holds semantic task payload constant across languages.
2. **Routing Architecture**:
   - **Do NOT route Indic traffic to a separate 6× capacity tier**.
   - Transition production serving stack to a unified multilingual model/tokenizer with a extended Indic vocabulary (e.g. XLM-RoBERTa, LLaMA-3.2, or Qwen2.5/Gemma2).
   - Capacity planning should allocate an Indic cost multiplier of **1.15× to 1.30× relative to English**, NOT 6.0×.

---

### Core Caveat & Production Monitoring

- **Biggest Caveat**: **Domain & Code-Mixing Skew**. Parallel formal evaluation corpora (FLORES/Samanantar) contain news/governmental text. In real production, user queries contain heavy English-Indic code-mixing (Hinglish/Kanglish) and informal colloquialisms. Code-mixing reduces effective Indic subword matching if words are written in Latin script or unstandardized spellings.
- **Production Metric to Monitor**: **`production_mean_tokens_per_request` by language tag / script**. Track the 7-day rolling average of total prompt + generated tokens per completed user request in production. If `mean_tokens(Indic) / mean_tokens(English)` exceeds **1.35×**, trigger a vocabulary re-tokenization review.
