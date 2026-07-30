#include "dual-deadline.cuh"

#include "ggml-backend-impl.h"
#include "ggml-cuda.h"

#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

// see dual-deadline.cuh for an overview and the paper reference

struct dd_config {
    bool enabled    = false;
    bool monolithic = false; // single deadline: all components copied before compute (A/B baseline)
    int  cache_slots = 32;   // per-tensor expert cache capacity
    int  max_tokens  = 4;    // largest micro-batch handled by this path
    bool stats      = false;
    bool validate   = false; // recompute every handled op via the standard path and compare
};

static const dd_config & dd_cfg() {
    static const dd_config cfg = [] {
        dd_config c;
        const char * env = getenv("GGML_CUDA_DUAL_DEADLINE");
        c.enabled = env != nullptr && atoi(env) != 0;
        if (!c.enabled) {
            return c;
        }
        const char * mode = getenv("GGML_CUDA_DD_MODE");
        c.monolithic = mode != nullptr && strcmp(mode, "mono") == 0;
        const char * slots = getenv("GGML_CUDA_DD_CACHE_EXPERTS");
        if (slots != nullptr && atoi(slots) > 0) {
            c.cache_slots = atoi(slots);
        }
        const char * toks = getenv("GGML_CUDA_DD_MAX_TOKENS");
        if (toks != nullptr && atoi(toks) > 0) {
            c.max_tokens = atoi(toks);
        }
        const char * st = getenv("GGML_CUDA_DD_STATS");
        c.stats = st != nullptr && atoi(st) != 0;
        const char * val = getenv("GGML_CUDA_DD_VALIDATE");
        c.validate = val != nullptr && atoi(val) != 0;
        // stderr directly: the ggml log callback may be nulled (llama-bench)
        // or already torn down (atexit stats) when these diagnostics fire
        fprintf(stderr, "dual-deadline: expert streaming enabled (%s, cache %d experts/tensor, max tokens %d%s%s)\n",
                c.monolithic ? "monolithic" : "staged", c.cache_slots, c.max_tokens,
                c.stats ? ", stats" : "", c.validate ? ", validate" : "");
        return c;
    }();
    return cfg;
}

bool ggml_cuda_dd_enabled() {
    return dd_cfg().enabled;
}

// persistent per-weight-tensor expert slice cache in device memory
struct dd_cache {
    char *   base        = nullptr;
    size_t   slice_bytes = 0;
    int      n_slots     = 0;
    int64_t  n_experts   = 0;
    std::vector<int32_t>  slot_of_expert; // expert id -> slot or -1
    std::vector<int32_t>  expert_of_slot; // slot -> expert id or -1
    std::vector<uint64_t> slot_tick;      // slot -> last-use tick for LRU
    int32_t * ids_dev    = nullptr;       // slot-remapped ids, device memory
    int32_t * ids_pinned = nullptr;       // pinned staging for ids upload
    size_t    ids_cap    = 0;             // capacity of the two ids buffers, in elements
    // stats
    uint64_t hits = 0, misses = 0;
    uint64_t bytes_copied = 0;
};

// per-node state for the current graph execution
struct dd_member {
    ggml_tensor * node    = nullptr;      // the MUL_MAT_ID dst node
    dd_cache *    cache   = nullptr;
    cudaEvent_t   ready   = nullptr;      // copies for this node complete
    bool          is_down = false;        // second-deadline component
};

struct dd_group {
    std::vector<dd_member> members;       // in graph order
    const ggml_tensor * ids = nullptr;
    bool staged   = false;
    bool fallback = false;
};

struct dd_device_state {
    uint64_t tick = 0;
    cudaStream_t copy_stream = nullptr;
    std::vector<cudaEvent_t> events;      // grows on demand, reused across graphs
    size_t next_event = 0;
    std::unordered_map<const void *, dd_cache> caches;         // key: weight data pointer
    std::unordered_map<const ggml_tensor *, size_t> node_group; // per-graph: node -> groups index
    std::vector<dd_group> groups;                               // per-graph
    // stats
    uint64_t groups_staged = 0, fallbacks = 0, validated = 0;
    double   max_validate_err = 0.0;
};

static dd_device_state & dd_state() {
    static dd_device_state state;
    return state;
}

static void dd_print_stats() {
    dd_device_state & st = dd_state();
    uint64_t hits = 0, misses = 0, bytes = 0;
    for (const auto & kv : st.caches) {
        hits   += kv.second.hits;
        misses += kv.second.misses;
        bytes  += kv.second.bytes_copied;
    }
    fprintf(stderr, "dual-deadline: %" PRIu64 " groups staged, %" PRIu64 " fallbacks, "
                    "cache hits %" PRIu64 " misses %" PRIu64 " (%.1f%% hit rate), %.1f MiB copied\n",
                    st.groups_staged, st.fallbacks, hits, misses,
                    hits + misses > 0 ? 100.0 * (double) hits / (double) (hits + misses) : 0.0,
                    (double) bytes / (1024.0 * 1024.0));
    if (dd_cfg().validate) {
        fprintf(stderr, "dual-deadline: %" PRIu64 " ops validated against the standard path, max abs err %.3e\n",
                        st.validated, st.max_validate_err);
    }
}

static cudaEvent_t dd_next_event() {
    dd_device_state & st = dd_state();
    if (st.next_event == st.events.size()) {
        cudaEvent_t ev;
        CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        st.events.push_back(ev);
    }
    return st.events[st.next_event++];
}

bool ggml_cuda_dd_should_handle(const ggml_tensor * dst) {
    if (!dd_cfg().enabled || dst->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }
    const ggml_tensor * src0 = dst->src[0];
    return src0 != nullptr && src0->buffer != nullptr &&
           src0->buffer->buft == ggml_backend_cuda_host_buffer_type();
}

void ggml_cuda_dd_prescan(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph) {
    GGML_UNUSED(ctx);
    dd_device_state & st = dd_state();
    st.node_group.clear();
    st.groups.clear();
    st.next_event = 0;

    if (dd_cfg().stats) {
        static bool registered = [] {
            atexit(dd_print_stats);
            return true;
        }();
        GGML_UNUSED(registered);
    }

    std::unordered_map<const ggml_tensor *, size_t> group_of_ids;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (!ggml_cuda_dd_should_handle(node)) {
            continue;
        }
        const ggml_tensor * ids = node->src[2];
        auto it = group_of_ids.find(ids);
        if (it == group_of_ids.end()) {
            it = group_of_ids.emplace(ids, st.groups.size()).first;
            st.groups.push_back({});
            st.groups.back().ids = ids;
        }
        dd_member m;
        m.node    = node;
        m.is_down = strstr(node->src[0]->name, "ffn_down") != nullptr;
        st.groups[it->second].members.push_back(m);
        st.node_group.emplace(node, it->second);
    }

    // a group is only stageable across two deadlines if exactly one member is
    // a down projection; otherwise treat all members as one deadline
    for (dd_group & g : st.groups) {
        int n_down = 0;
        for (const dd_member & m : g.members) {
            n_down += m.is_down ? 1 : 0;
        }
        if (n_down != 1) {
            for (dd_member & m : g.members) {
                m.is_down = false;
            }
        }
    }
}

static dd_cache * dd_get_cache(const ggml_tensor * src0, int64_t n_ids_needed) {
    dd_device_state & st = dd_state();
    dd_cache & cache = st.caches[src0->data];
    if (cache.base == nullptr) {
        cache.slice_bytes = src0->nb[2];
        cache.n_experts   = src0->ne[2];
        cache.n_slots     = (int) std::min<int64_t>(dd_cfg().cache_slots, cache.n_experts);
        // small tail padding so quantized kernels may safely over-read the last slice;
        // the memset must be ordered on the copy stream: the legacy default stream
        // does not synchronize with the non-blocking copy stream, and an unordered
        // memset races with (and can clobber) the first slice copies
        CUDA_CHECK(cudaMalloc(&cache.base, cache.slice_bytes * cache.n_slots + 512));
        CUDA_CHECK(cudaMemsetAsync(cache.base, 0, cache.slice_bytes * cache.n_slots + 512, st.copy_stream));
        cache.slot_of_expert.assign(cache.n_experts, -1);
        cache.expert_of_slot.assign(cache.n_slots, -1);
        cache.slot_tick.assign(cache.n_slots, 0);
    }
    if (cache.ids_cap < (size_t) n_ids_needed) {
        if (cache.ids_dev != nullptr) {
            CUDA_CHECK(cudaFree(cache.ids_dev));
            CUDA_CHECK(cudaFreeHost(cache.ids_pinned));
        }
        cache.ids_cap = n_ids_needed;
        CUDA_CHECK(cudaMalloc(&cache.ids_dev, cache.ids_cap * sizeof(int32_t)));
        CUDA_CHECK(cudaMallocHost(&cache.ids_pinned, cache.ids_cap * sizeof(int32_t)));
    }
    return &cache;
}

// ensure the given experts are resident in the cache, enqueueing slice copies
// on the copy stream; returns false if the working set exceeds the cache
static bool dd_ensure_experts(dd_cache & cache, const ggml_tensor * src0,
                              const std::vector<int32_t> & experts, cudaStream_t copy_stream) {
    dd_device_state & st = dd_state();
    if ((int64_t) experts.size() > cache.n_slots) {
        return false;
    }
    ++st.tick;
    // first pass: mark hits so their slots cannot be chosen as victims
    for (int32_t e : experts) {
        const int32_t slot = cache.slot_of_expert[e];
        if (slot >= 0) {
            cache.slot_tick[slot] = st.tick;
            ++cache.hits;
        }
    }
    for (int32_t e : experts) {
        if (cache.slot_of_expert[e] >= 0) {
            continue;
        }
        // LRU victim
        int victim = 0;
        for (int s = 1; s < cache.n_slots; ++s) {
            if (cache.slot_tick[s] < cache.slot_tick[victim]) {
                victim = s;
            }
        }
        GGML_ASSERT(cache.slot_tick[victim] != st.tick); // guaranteed by size check above
        if (cache.expert_of_slot[victim] >= 0) {
            cache.slot_of_expert[cache.expert_of_slot[victim]] = -1;
        }
        cache.expert_of_slot[victim] = e;
        cache.slot_of_expert[e]      = victim;
        cache.slot_tick[victim]      = st.tick;
        CUDA_CHECK(cudaMemcpyAsync(cache.base + (size_t) victim * cache.slice_bytes,
                                   (const char *) src0->data + (size_t) e * src0->nb[2],
                                   cache.slice_bytes, cudaMemcpyHostToDevice, copy_stream));
        ++cache.misses;
        cache.bytes_copied += cache.slice_bytes;
    }
    return true;
}

// copy the ids for one member, remapped from expert ids to cache slots
static void dd_upload_remapped_ids(dd_cache & cache, const std::vector<int32_t> & ids_host,
                                   cudaStream_t copy_stream) {
    for (size_t i = 0; i < ids_host.size(); ++i) {
        cache.ids_pinned[i] = cache.slot_of_expert[ids_host[i]];
        GGML_ASSERT(cache.ids_pinned[i] >= 0);
    }
    CUDA_CHECK(cudaMemcpyAsync(cache.ids_dev, cache.ids_pinned, ids_host.size() * sizeof(int32_t),
                               cudaMemcpyHostToDevice, copy_stream));
}

// read the ids tensor to the host; synchronizes the compute stream, which also
// guarantees all previously enqueued reads of cache slots have completed, so
// evictions below cannot race with in-flight kernels
static void dd_read_ids(ggml_backend_cuda_context & ctx, const ggml_tensor * ids,
                        std::vector<int32_t> & out) {
    const int64_t n_used   = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    std::vector<char> raw(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(raw.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    out.resize(n_used * n_tokens);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t j = 0; j < n_used; ++j) {
            out[t * n_used + j] = *(const int32_t *) (raw.data() + t * ids->nb[1] + j * ids->nb[0]);
        }
    }
}

static bool dd_stage_group(ggml_backend_cuda_context & ctx, dd_group & g) {
    dd_device_state & st = dd_state();
    const dd_config & cfg = dd_cfg();

    if (st.copy_stream == nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&st.copy_stream, cudaStreamNonBlocking));
    }

    std::vector<int32_t> ids_host;
    dd_read_ids(ctx, g.ids, ids_host);

    // unique experts in first-seen order
    std::vector<int32_t> experts;
    for (int32_t e : ids_host) {
        bool seen = false;
        for (int32_t u : experts) {
            if (u == e) { seen = true; break; }
        }
        if (!seen) {
            experts.push_back(e);
        }
    }

    // copy order realizes the two deadlines: every first-deadline (gate/up)
    // component of the group is enqueued before any down component, and the
    // down copies overlap the gate/up computation that runs after ev_first
    cudaEvent_t ev_first = nullptr;
    dd_member * down     = nullptr;
    for (dd_member & m : g.members) {
        if (m.is_down) {
            down = &m;
            continue;
        }
        m.cache = dd_get_cache(m.node->src[0], (int64_t) ids_host.size());
        if (!dd_ensure_experts(*m.cache, m.node->src[0], experts, st.copy_stream)) {
            return false;
        }
        dd_upload_remapped_ids(*m.cache, ids_host, st.copy_stream);
    }
    if (!cfg.monolithic) {
        ev_first = dd_next_event();
        CUDA_CHECK(cudaEventRecord(ev_first, st.copy_stream));
    }
    if (down != nullptr) {
        down->cache = dd_get_cache(down->node->src[0], (int64_t) ids_host.size());
        if (!dd_ensure_experts(*down->cache, down->node->src[0], experts, st.copy_stream)) {
            return false;
        }
        dd_upload_remapped_ids(*down->cache, ids_host, st.copy_stream);
    }
    cudaEvent_t ev_last = dd_next_event();
    CUDA_CHECK(cudaEventRecord(ev_last, st.copy_stream));

    for (dd_member & m : g.members) {
        m.ready = (cfg.monolithic || m.is_down) ? ev_last : ev_first;
    }

    ++st.groups_staged;
    g.staged = true;
    return true;
}

static double dd_max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double max_err = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        max_err = std::max(max_err, (double) fabsf(a[i] - b[i]));
    }
    return max_err;
}

// compare the dual-deadline result against the standard path; both sides are
// computed twice so a mismatch can be attributed to whichever side is unstable
static void dd_validate(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                        ggml_tensor * src0_ws, ggml_tensor * ids_ws) {
    dd_device_state & st = dd_state();
    const size_t nbytes = ggml_nbytes(dst);
    ggml_cuda_pool_alloc<char> ref1(ctx.pool(), nbytes);
    ggml_cuda_pool_alloc<char> ref2(ctx.pool(), nbytes);
    ggml_cuda_pool_alloc<char> dd2 (ctx.pool(), nbytes);
    void * dd_data = dst->data;

    ggml_tensor * src0_orig = dst->src[0];
    ggml_tensor * ids_orig  = dst->src[2];

    dst->data = ref1.ptr;
    ggml_cuda_mul_mat_id(ctx, dst); // standard path, src0 in place in pinned host memory
    dst->data = ref2.ptr;
    ggml_cuda_mul_mat_id(ctx, dst);
    dst->src[0] = src0_ws;
    dst->src[2] = ids_ws;
    dst->data   = dd2.ptr;
    ggml_cuda_mul_mat_id(ctx, dst); // dual-deadline path again, from the workspace
    dst->src[0] = src0_orig;
    dst->src[2] = ids_orig;
    dst->data   = dd_data;
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));

    std::vector<float> v_dd(ggml_nelements(dst)), v_dd2(ggml_nelements(dst));
    std::vector<float> v_r1(ggml_nelements(dst)), v_r2 (ggml_nelements(dst));
    CUDA_CHECK(cudaMemcpy(v_dd.data(),  dd_data,  nbytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_dd2.data(), dd2.ptr,  nbytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_r1.data(),  ref1.ptr, nbytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_r2.data(),  ref2.ptr, nbytes, cudaMemcpyDeviceToHost));

    const double err_dd_ref  = dd_max_abs_diff(v_dd,  v_r1);
    const double err_dd_dd   = dd_max_abs_diff(v_dd,  v_dd2);
    const double err_ref_ref = dd_max_abs_diff(v_r1,  v_r2);

    ++st.validated;
    st.max_validate_err = std::max(st.max_validate_err, err_dd_ref);
    if (err_dd_ref != 0.0 || err_dd_dd != 0.0 || err_ref_ref != 0.0) {
        fprintf(stderr, "dual-deadline: %s mismatch: dd-vs-ref %.3e, dd-vs-dd %.3e, ref-vs-ref %.3e\n",
                        dst->name, err_dd_ref, err_dd_dd, err_ref_ref);

        // byte-compare every used expert's workspace slot against its host slice
        dd_cache & cache = st.caches[src0_orig->data];
        std::vector<int32_t> ids_host;
        dd_read_ids(ctx, ids_orig, ids_host);
        const size_t sb = cache.slice_bytes;
        std::vector<char> ws_bytes(sb), host_view(sb);
        for (size_t i = 0; i < ids_host.size(); ++i) {
            const int32_t e    = ids_host[i];
            const int32_t slot = cache.slot_of_expert[e];
            CUDA_CHECK(cudaMemcpy(ws_bytes.data(), (char *) src0_ws->data + (size_t) slot * sb, sb,
                                  cudaMemcpyDeviceToHost));
            memcpy(host_view.data(), (const char *) src0_orig->data + (size_t) e * src0_orig->nb[2], sb);
            const bool same = memcmp(ws_bytes.data(), host_view.data(), sb) == 0;
            fprintf(stderr, "dual-deadline:   ids[%zu]=expert %d -> slot %d, slice bytes %s\n",
                            i, e, slot, same ? "MATCH" : "DIFFER");
        }
    }
}

bool ggml_cuda_dd_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    dd_device_state & st = dd_state();
    const ggml_tensor * ids = dst->src[2];

    if (ids->ne[1] > dd_cfg().max_tokens) {
        return false; // larger batches read src0 in place via the standard path
    }

    auto it = st.node_group.find(dst);
    if (it == st.node_group.end()) {
        return false;
    }
    dd_group & g = st.groups[it->second];
    if (g.fallback) {
        return false;
    }
    if (!g.staged && !dd_stage_group(ctx, g)) {
        g.fallback = true;
        ++st.fallbacks;
        return false;
    }

    dd_member * m = nullptr;
    for (dd_member & cand : g.members) {
        if (cand.node == dst) {
            m = &cand;
            break;
        }
    }
    GGML_ASSERT(m != nullptr && m->cache != nullptr);

    CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), m->ready, 0));

    // shadow tensors: same op on the compact slot workspace with remapped ids;
    // buffer is pointed at a device buffer so downstream dispatch predicates
    // treat the sources as device-resident
    ggml_tensor src0_ws = *dst->src[0];
    src0_ws.data   = m->cache->base;
    src0_ws.ne[2]  = m->cache->n_slots;
    src0_ws.nb[3]  = src0_ws.nb[2] * m->cache->n_slots;
    src0_ws.buffer = dst->buffer;

    ggml_tensor ids_ws = *ids;
    ids_ws.data   = m->cache->ids_dev;
    ids_ws.nb[0]  = sizeof(int32_t);
    ids_ws.nb[1]  = ids->ne[0] * sizeof(int32_t);
    ids_ws.nb[2]  = ids_ws.nb[1] * ids->ne[1];
    ids_ws.nb[3]  = ids_ws.nb[2];
    ids_ws.buffer = dst->buffer;

    ggml_tensor * src0_orig = dst->src[0];
    ggml_tensor * ids_orig  = dst->src[2];
    dst->src[0] = &src0_ws;
    dst->src[2] = &ids_ws;
    ggml_cuda_mul_mat_id(ctx, dst);
    dst->src[0] = src0_orig;
    dst->src[2] = ids_orig;

    if (dd_cfg().validate) {
        dd_validate(ctx, dst, &src0_ws, &ids_ws);
    }
    return true;
}
