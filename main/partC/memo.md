# Strategic Recommendation: Casual Indic Conversational Persona

**To:** Head of Product & AI Engineering  
**From:** AI Team Audit Taskforce  
**Date:** September 6, 2026  
**Subject:** Technical Path & Strategy for Casual Indic Assistant Responses  

---

### Executive Recommendation

We recommend **Path (c): Prompt-Engineering Only (Few-Shot Persona Systems)** as our primary path, backed by a strict Day-7 Kill Criterion triggering **Path (a) QLoRA SFT** if prompt compliance fails.

Path (b) (1B Rewriter Model) is rejected due to a $+1.5\text{s}$ latency tax, double KV cache memory overhead, and severe resource contention on serving infrastructure.

---

### Explicitly Labelled Decision Framework

#### 1. Assumptions
- **Language Coverage & Human Bottleneck**: Our sole reviewer covers Hindi and Kannada only (10 hrs/week). Tamil, Telugu, Bengali, and Marathi have **zero native human coverage**. Any path chosen must minimize un-evaluated model weight changes that risk alignment drift in un-reviewed languages.
- **Resource Limits**: 1× A100-80GB GPU for 2 weeks; $0 external API budget (all local inference/generation); 3-week launch deadline.
- **Inference Constraints**: Added latency per request must remain $< 100\text{ ms}$ TTFT.

#### 2. Back-of-Envelope Arithmetic
- **Reviewer Throughput Budget**:
  - Evaluation speed: $\sim 15 \text{ responses/hour}$ (blind side-by-side pairwise scoring).
  - Total reviewer capacity over 3 weeks: $3 \times 10 \text{ hrs} = 30 \text{ hours} \implies 450 \text{ total evaluated responses}$ ($200$ Hindi, $200$ Kannada, $50$ calibration).
  - An evaluation round of 100 responses (50 Hindi / 50 Kannada) requires $6.6 \text{ reviewer hours}$. We can afford exactly **4 iteration cycles**.
- **Latency & Compute Cost Comparison**:
  - *Path (c) Few-Shot Prompting*: Adds $\sim 150$ prompt tokens (system persona + 3 few-shot pairs). Prefill time on A100 increases by $< 40\text{ ms}$; decode latency and memory are **$0\%$ extra cost**.
  - *Path (b) 1B Rewriter*: Requires 2nd full decode pass ($200$ gen tokens). Adds $+1.8\text{s}$ decode latency, doubles KV cache consumption, and reduces overall serving capacity by $\sim 50\%$.
  - *Path (a) Local SFT*: Fine-tuning an 8B model via QLoRA on A100 takes $\sim 4 \text{ hours}$ per run. Local synthetic data generation ($10,000$ responses @ $50 \text{ tok/s}$) takes $11.1 \text{ hours}$, but risks noisy synthetic target quality without external teacher APIs.

#### 3. Success Metric & Numeric Threshold
- **Primary Metric**: **Casualness Win-Rate** in blind side-by-side pairwise evaluation against baseline formal outputs on native reviewer test sets (Hindi & Kannada).
- **Target Threshold**: **$\ge 75\%$ Casualness Win-Rate** with **$\le 2.0\%$ degradation in factual accuracy or safety score**.

#### 4. Kill Criterion & Timeline
- **Observation**: If by **End of Day 7 (Week 1)**, Few-Shot System Prompting fails to achieve **$\ge 60\%$ Casualness Win-Rate** on Hindi/Kannada OR exhibits **$> 5\%$ non-compliance/hallucination rate** (e.g., mixing formal verbs despite casual prompt instructions).
- **Action**: Immediately pivot on Day 8 to **Path (a) QLoRA SFT**. Use the A100 GPU throughout Week 2 to fine-tune a 3B/8B model using 2,000 hand-curated and translated casual seed pairs across all 6 languages.

#### 5. First Experiment on Day 1
- **Experiment Setup**: Create a benchmark set of 20 representative prompts (10 Hindi, 10 Kannada) spanning greetings, recommendations, troubleshooting, and opinion queries.
- **Execution**: Run local A100 inference comparing Baseline (Formal) against 3 candidate System Prompt personas:
  1. *Prompt P1*: Explicit persona instruction ("Speak in warm, everyday conversational Hindi/Kannada...").
  2. *Prompt P2*: Persona instruction + 3 curated Few-Shot casual dialogue exemplars.
  3. *Prompt P3*: Colloquial dialect / region-specific vocabulary markers.
- **Validation**: Deliver the 20 paired outputs to the native reviewer on Day 2 for an initial 2-hour baseline calibration session.
