# Part B — Capacity Reconciliation & Serving Load Test Audit

**Author:** AI Team Audit Taskforce  
**Reference Files:** `bench/model_spec.md`, `bench/bench_log.csv`  

---

## B1. Theoretical Capacity Derivation & Log Verification

### Model Parameters (FLM-4B-Instruct dense)
- Layers ($L$): $28$
- Attention heads ($Q$): $24$
- KV heads ($H_{kv}$): $8$ (GQA 3:1 ratio)
- Head dimension ($d_k$): $128$
- Precision: FP16 ($2 \text{ bytes/element}$)
- GPU VRAM: $24 \text{ GB}$ (NVIDIA L4)
- `gpu_memory_utilization`: $0.92$
- Non-KV runtime overhead: ~ $1.60 \text{ GB}$

---

### (a) Exact KV-Cache Bytes per Token

For each token at each layer, the KV cache stores a Key vector ($d_k = 128$) and a Value vector ($d_k = 128$) across $H_{kv} = 8$ heads in FP16 precision ($2 \text{ bytes}$ per element):

$$\text{Key cache per layer} = H_{kv} \times d_k \times \text{bytes\_per\_elem} = 8 \times 128 \times 2 = 2,048 \text{ bytes}$$

$$\text{Value cache per layer} = H_{kv} \times d_k \times \text{bytes\_per\_elem} = 8 \times 128 \times 2 = 2,048 \text{ bytes}$$

$$\text{Total KV cache per layer} = 2,048 + 2,048 = 4,096 \text{ bytes} = 4 \text{ KiB}$$

Across all $L = 28$ layers:

$$\text{KV bytes per token} = 28 \times 4,096 \text{ bytes} = 114,688 \text{ bytes} = 112 \text{ KiB} = 0.114688 \text{ MB}$$

---

### (b) Maximum Concurrent 4096-Token Sequences

1. **Usable VRAM ($92\%$ allocation)**:
   $$VRAM_{usable} = 0.92 \times 24 \text{ GB} = 22.08 \text{ GB} = 22,080,000,000 \text{ bytes}$$

2. **Model Weights Memory**:
   $$M_{weights} = 4.2 \times 10^9 \text{ params} \times 2 \text{ bytes/param} = 8.40 \text{ GB} = 8,400,000,000 \text{ bytes}$$

3. **Non-KV Runtime Overhead** (activations, CUDA graphs, workspace):
   $$M_{overhead} = 1.60 \text{ GB} = 1,600,000,000 \text{ bytes}$$

4. **Available KV Cache Memory Pool ($M_{KV\_pool}$)**:
   $$M_{KV\_pool} = 22.08 \text{ GB} - 8.40 \text{ GB} - 1.60 \text{ GB} = 12.08 \text{ GB} = 12,080,000,000 \text{ bytes}$$
   *(In binary GiB: $25.77 \text{ GiB} \times 0.92 - 7.82 \text{ GiB} - 1.49 \text{ GiB} = 13.71 \text{ GB} = 12.77 \text{ GiB}$)*

5. **KV Cache per 4096-token sequence ($KV_{seq}$)**:
   $$KV_{seq} = 4096 \text{ tokens} \times 114,688 \text{ bytes/token} = 469,762,048 \text{ bytes} = 0.46976 \text{ GB} = 0.4375 \text{ GiB}$$

6. **Maximum Concurrent Sequences ($N_{max}$)**:
   $$N_{max} = \left\lfloor \frac{M_{KV\_pool}}{KV_{seq}} \right\rfloor = \left\lfloor \frac{12.08 \text{ GB}}{0.46976 \text{ GB}} \right\rfloor = \lfloor 25.71 \rfloor = \mathbf{25 \text{ sequences}}$$
   *(Binary GiB bounds: $12.77 \text{ GiB} / 0.4375 \text{ GiB} = \mathbf{29 \text{ sequences}}$)*

---

### Verification Against `bench_log.csv`

In `bench_log.csv` for long prompt requests (prompt 3584 + gen 512 = 4096 total sequence length):
- **Batch 24**: `kv_cache_util = 0.93`, `preempted_seqs = 0`. All 24 sequences fit within KV memory.
- **Batch 32**: `kv_cache_util = 0.97`, `preempted_seqs = 7`. Preemption begins because 32 sequences exceed the max memory capacity of ~25–29 sequences.
- **Batch 48**: `kv_cache_util = 0.97`, `preempted_seqs = 23`. Severe memory thrashing and block preemption.

The theoretical upper bound of **25–29 sequences** perfectly predicts the onset of preemption between batch 24 and batch 32!

---

## B2. Long-Context Throughput Anomaly Analysis

### The Anomaly
In the long-context sweep (prompt 3584, gen 512), throughput scales up to batch 24 (1607.4 tok/s), but **drops** to 1384.0 tok/s at batch 32 and **collapses** to 1298.5 tok/s at batch 48. This contradicts naive batching assumptions that throughput scales monotonically with batch size.

### Underlying Mechanism
1. **Memory Exhaustion**: Requiring $32 \times 469.76 \text{ MB} = 15.03 \text{ GB}$ of KV cache exceeds the GPU's $12.08 \text{ GB}$ KV pool capacity.
2. **Preemption Thrashing**: At batch 32, vLLM's block manager runs out of free KV blocks (`kv_cache_util = 0.97`), forcing the scheduler to **preempt 7 sequences** by evicting their KV blocks. At batch 48, **23 sequences are preempted**.
3. **Prefill Re-Execution Penalty**: When preempted sequences resume, their entire 3,584-token prompt must be re-computed from scratch. This prefill re-execution wastes massive GPU compute, driving `ttft_ms_p50` from 500.5ms to 955.4ms and `e2e_ms_p95` from 69.2s to 105.4s.

### Proposed Config Change & Predicted Quantitative Effect
- **Config Change**: Set `max_num_seqs = 24` in the serving engine configuration (or cap max batch size in the scheduler/load-balancer to 24).
- **Predicted Effect**:
  - **Preemptions**: Reduced from 23 to **0**.
  - **Throughput**: Maintained at peak **1607.4 tok/s** (a **+23.8% improvement** over batch 48's degraded 1298.5 tok/s).
  - **End-to-End Latency (p95)**: Reduced from 105.4s to **69.2s** (a **34.4% reduction**).
  - **TTFT (p50)**: Reduced from 955.4ms to **500.5ms** (a **47.6% reduction**).

---

## B3. REPORT_v0 Misreading & Honest Goodput Derivation

### The Misreading
`REPORT_v0` Section 2 misread `reported_tok_s` as output generation throughput. In benchmark harnesses, `reported_tok_s` is calculated as:

$$\text{reported\_tok\_s} = \frac{\text{num\_requests} \times (\text{prompt\_len} + \text{gen\_len})}{\text{wall\_clock\_s}}$$

For long prompts (3584 prompt + 512 gen = 4096 total tokens), prompt prefill tokens make up **87.5%** of all tokens. Because prefill processes all 3,584 prompt tokens in parallel in a single forward pass, `reported_tok_s` is heavily inflated by prompt tokens. `REPORT_v0` incorrectly concluded that longer prompts yield better throughput and naively projected batch 48 to ~3200 tok/s.

---

### Honest Goodput Derivation (Batch 24, Prompt 3584)

#### Method 1: Generated Output Tokens / Decode Wall-Clock Time
- Total generated tokens = $24 \text{ requests} \times 512 \text{ gen tokens} = 12,288 \text{ gen tokens}$
- Total wall clock = $61.16 \text{ seconds}$
- Prefill time (TTFT p50) = $500.5 \text{ ms} = 0.5005 \text{ seconds}$
- Decode wall clock = $61.16 - 0.5005 = 60.6595 \text{ seconds}$

$$\text{Decode Goodput} = \frac{12,288 \text{ gen tokens}}{60.6595 \text{ seconds}} = \mathbf{202.57 \text{ gen tok/s}}$$
*(Overall end-to-end output rate: $12,288 / 61.16 = 200.92 \text{ gen tok/s}$)*

#### Method 2: Inverse of Inter-Token Latency (ITL)
- Median ITL during decode (`itl_ms_p50`) = $96.07 \text{ ms} = 0.09607 \text{ seconds/token}$
- Decode generation speed per sequence = $\frac{1}{0.09607 \text{ s}} = 10.409 \text{ gen tok/s/user}$
- Across 24 active concurrent requests:

$$\text{Decode Goodput} = 24 \times 10.409 = \frac{24}{0.09607 \text{ s}} = \mathbf{249.82 \text{ gen tok/s}}$$

---

### Corrected Report Summary
"Longer prompts increase prefill token count and compute saturation, inflating reported aggregate throughput (`reported_tok_s`), but actual user-facing generation throughput ('goodput') is ~200–250 gen tok/s. Furthermore, batch sizes above 24 cause KV cache exhaustion and preemption thrashing, reducing throughput to 1298 tok/s at batch 48 rather than scaling linearly to 3200 tok/s. Batch capacity must be capped at 24."

---

## B4. Telemetry Counter for Preemption Verification

- **Primary Metric / Counter**: `vllm:num_preemptions_total` (or `vllm:gpu_cache_usage_perc` / `vllm:cpu_swap_space_usage`).
- **Expected Value**: `vllm:num_preemptions_total` = **0** for batch size $\le 24$; jumps to **7** at batch 32 and **23** at batch 48, while `vllm:gpu_cache_usage_perc` reaches **97%**.
