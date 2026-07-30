#pragma once

// DualDeadline: component-staged expert streaming for host-resident MoE expert
// weights, after:
//
//   Marcel Butucea, "DualDeadline: Component-Staged Expert Prefetching for
//   Exact Offloaded Mixture-of-Experts Inference", 2026.
//   https://github.com/unixsysdev/dualdeadline
//
// A gated MoE expert has no single transfer deadline: gate/up projections are
// needed when expert computation starts, but the down projection is not
// consumed until gate/up arithmetic has produced the intermediate activation.
// When expert tensors live in (pinned) host memory, this module copies only
// the router-selected expert slices to a persistent per-tensor device cache,
// ordered gate/up first, and overlaps the down-projection copies with gate/up
// computation on a dedicated copy stream. Routing and results are exact:
// prediction is not involved, misses are simply copied before use.
//
// Enabled with GGML_CUDA_DUAL_DEADLINE=1. See docs/dual-deadline.md.

#include "common.cuh"

// true when GGML_CUDA_DUAL_DEADLINE=1 is set in the environment
bool ggml_cuda_dd_enabled();

// true if this MUL_MAT_ID node has a host-resident src0 that the
// dual-deadline path can take over
bool ggml_cuda_dd_should_handle(const ggml_tensor * dst);

// group gate/up/down MUL_MAT_ID nodes that share an ids tensor and reset
// per-execution state; call once per graph execution before evaluating nodes
void ggml_cuda_dd_prescan(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph);

// execute a MUL_MAT_ID node via the dual-deadline path;
// returns false when the node cannot be handled (caller must fall back)
bool ggml_cuda_dd_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// standard MUL_MAT_ID entry point in ggml-cuda.cu (used for compute and for
// GGML_CUDA_DD_VALIDATE=1 reference checks)
void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
