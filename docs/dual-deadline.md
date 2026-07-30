# DualDeadline: component-staged expert streaming for offloaded MoE

This branch implements **DualDeadline**, a component-staged transfer schedule
for mixture-of-experts models whose expert tensors are kept in host memory,
after:

> Marcel Butucea, *DualDeadline: Component-Staged Expert Prefetching for Exact
> Offloaded Mixture-of-Experts Inference*, 2026.
> https://github.com/unixsysdev/dualdeadline

## Idea

A gated MoE expert consists of three matrices. The `gate` and `up` projections
both consume the layer input, but the `down` projection is not needed until
gate/up arithmetic has produced the intermediate activation:

```
y_e = W_down [ SiLU(W_gate x) ⊙ W_up x ]
```

An expert therefore has **two transfer deadlines**, not one. When expert
weights live in host memory, this implementation:

1. reads the routed expert ids as soon as the router has produced them;
2. copies only the **selected** experts' `gate`/`up` slices to a persistent
   per-tensor device cache (LRU over a configurable number of expert slots),
   on a dedicated copy stream;
3. starts gate/up computation as soon as those copies complete, while the
   `down` slices of the *same* experts are still being copied — the second
   deadline's transfers overlap the first deadline's compute;
4. runs the down matmul once its copies complete.

Routing and results are **exact**. There is no prediction and no approximation
in this implementation: the native router output determines every copy, and a
validation mode cross-checks results against the standard execution path.

## Relationship to upstream

Upstream llama.cpp already contains two related mechanisms:

- the scheduler copies **only the used experts** of host-resident MoE weights
  when an offloaded split begins (`ggml_backend_sched_compute_splits`), but it
  is monolithic (all three projections before compute), transient (no reuse
  across tokens), and only active for batches ≥ 32 (prompt processing);
- on **integrated GPUs**, host-resident weights are read in place, zero-copy.

DualDeadline adds: batch-1 decode support, the two-deadline staged schedule
with copy/compute overlap, and a persistent cross-token expert cache.

## Usage

Expert tensors must reside in pinned host memory, which is what `--cpu-moe`
(or `--n-cpu-moe N`, or `-ot 'exps=CPU'`) already produces when combined with
`--no-mmap`:

```sh
GGML_CUDA_DUAL_DEADLINE=1 \
llama-cli -m model.gguf -ngl 999 --cpu-moe --no-mmap -p "..."
```

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `GGML_CUDA_DUAL_DEADLINE` | off | `1` enables the path |
| `GGML_CUDA_DD_MODE` | `staged` | `mono` copies all components before compute (single-deadline A/B baseline) |
| `GGML_CUDA_DD_CACHE_EXPERTS` | 32 | expert slots per weight tensor in the device cache |
| `GGML_CUDA_DD_MAX_TOKENS` | 4 | largest micro-batch handled; larger batches use the standard path |
| `GGML_CUDA_DD_STATS` | off | `1` prints staging/cache statistics at exit |
| `GGML_CUDA_DD_VALIDATE` | off | `1` recomputes every handled op via the standard path and compares |

`GGML_CUDA_DD_MODE=mono` exists purely to reproduce the paper's
monolithic-versus-staged comparison inside a real decode loop: identical
selective copies and cache, single completion event instead of two.

## Measured results

Qwen3.6-35B-A3B (Q4_K_XL, 256 experts/layer, top-8), all expert tensors in
pinned host memory, AMD Ryzen AI MAX+ 395 / Radeon 8060S (Strix Halo, unified
memory), ROCm 7.2.4, `llama-bench --load-mode none -ncmoe 999 -t 16 -r 3`:

| configuration | tg64 t/s |
|---|---:|
| upstream baseline: expert matmuls on CPU, 16 threads | 49.7 ± 0.3 |
| staged, cache 64 experts/tensor (84% hit rate) | 44.1 ± 0.4 |
| monolithic, cache 64 | 42.7 ± 0.4 |
| staged, cache 32 | 41.6 ± 0.4 |
| monolithic, cache 32 | 39.6 ± 0.4 |

Staged beats monolithic by 3–5% end-to-end at identical copies, cache, and
kernels — the two-deadline effect in isolation. The CPU baseline wins overall
on this machine because unified memory offers the GPU no transfer to hide;
see the paper for the regime analysis. Exactness was verified with
`GGML_CUDA_DD_VALIDATE=1`: 3,120 intercepted matmuls recomputed via the
standard path with maximum absolute error 0.0.

## Scope and limitations

This is a research prototype:

- Targeted at unified-memory APUs (developed and validated on an AMD Strix
  Halo / Radeon 8060S with ROCm) and at discrete GPUs where MoE experts do
  not fit in VRAM. On discrete hardware the same code path applies but has
  not been benchmarked here.
- On a UMA APU, host and device memory share DRAM bandwidth, so explicit
  staging competes with zero-copy in-place reads; the interesting comparison
  on such hardware is `staged` versus `mono` at equal copies, which isolates
  the value of the second deadline. On PCIe-attached GPUs the copy engine
  crosses a real link and the staged schedule additionally replaces CPU-side
  expert computation.
- Speculative gate/up prefetch before the router fires (the learned-predictor
  part of the paper) is **not** implemented; the cross-token LRU cache plays
  the role of training-free temporal-locality "prediction". The staging
  machinery is predictor-ready: a predictor would simply enqueue gate/up
  copies for its candidate set before the ids are known.
- The device cache is process-lifetime and is not returned when a model is
  unloaded.
- CUDA graph capture is disabled while the path is active (reading the routed
  ids requires a stream synchronization per MoE layer).
- Enabling `GGML_CUDA_DUAL_DEADLINE=1` also makes the CUDA backend claim ops
  on pinned-host weights in place (as it already does on integrated GPUs), so
  batches above `GGML_CUDA_DD_MAX_TOKENS` read host memory directly from
  device kernels. On UMA this is native; on discrete GPUs it crosses PCIe.

## Files

- `ggml/src/ggml-cuda/dual-deadline.cuh` / `dual-deadline.cu` — the mechanism
- `ggml/src/ggml-cuda/ggml-cuda.cu` — dispatch hook, fusion guards, pinned
  host buffer acceptance, CUDA-graph guard
