// TurboQuant tq3_0 MMA instances - valid Ampere ncols: 8,16,32,64
#include "../fattn-mma-tq3_0.cuh"

template<int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_tq3_0_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    constexpr int ncols = ncols1 * ncols2;
    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int  nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, cc);
    const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);
    const int  nstages        = ggml_cuda_fattn_mma_get_nstages       (DKQ, DV, ncols1, ncols2, cc);
    const int cols_per_warp  = std::min(ncols, get_cols_per_warp(cc));
    const int warp_size_host = ggml_cuda_info().devices[ctx.device].warp_size;
    const int nwarps         = nthreads / warp_size_host;
    constexpr bool V_is_K_view = false;

    const size_t nbytes_shared_KV_1stage = nbatch_fa            * std::max(nbatch_K2 + 4,  nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_KV_2stage = nbatch_fa            *         (nbatch_K2 + 4 + nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_Q         = ncols                * (DKQ/2 + 4)                             * sizeof(half2);
    const size_t nbytes_shared_mask      = ncols1               * (nbatch_fa/2 + 4)                       * sizeof(half2);
    const size_t nbytes_shared_combine   = nwarps*cols_per_warp * (nbatch_combine + 4)                    * sizeof(half2);
    const size_t nbytes_shared_KV        = nstages <= 1 ? nbytes_shared_KV_1stage : nbytes_shared_KV_2stage;
    const size_t nbytes_shared_total     = std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q,  nbytes_shared_KV + nbytes_shared_mask) :
                 nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    const ggml_tensor * V_tensor = dst->src[2];
    const bool v_is_tq3_0v = V_tensor->type == GGML_TYPE_TQ3_0V;
    using fattn_kernel_ptr_t = fattn_kernel_t;
    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = v_is_tq3_0v ?
            (fattn_kernel_t)flash_attn_ext_tq3_0<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, true> :
            (fattn_kernel_t)flash_attn_ext_tq3_0<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, false>;
        static bool smem_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!smem_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel),
                cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            smem_raised[id] = true;
        }
        } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = v_is_tq3_0v ?
            (fattn_kernel_t)flash_attn_ext_tq3_0<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, true> :
            (fattn_kernel_t)flash_attn_ext_tq3_0<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, false>;
        static bool smem_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!smem_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel),
                cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            smem_raised[id] = true;
        }
    }
    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, false, !v_is_tq3_0v, true, warp_size_host);
}


DECL_FATTN_MMA_TQ3_0_CASE(128, 128,  8, 1);
DECL_FATTN_MMA_TQ3_0_CASE(128, 128, 16, 1);
DECL_FATTN_MMA_TQ3_0_CASE(128, 128, 32, 1);
DECL_FATTN_MMA_TQ3_0_CASE(128, 128, 64, 1);
DECL_FATTN_MMA_TQ3_0_CASE(256, 256,  8, 1);
DECL_FATTN_MMA_TQ3_0_CASE(256, 256, 16, 1);
DECL_FATTN_MMA_TQ3_0_CASE(256, 256, 32, 1);
DECL_FATTN_MMA_TQ3_0_CASE(256, 256, 64, 1);
