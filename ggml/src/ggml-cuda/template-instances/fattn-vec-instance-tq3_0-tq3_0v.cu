// TurboQuant tq3_0 K-cache with tq3_0v V-cache (original space, no WHT dequant)
#include "../fattn-vec.cuh"
DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TQ3_0, GGML_TYPE_TQ3_0V);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_TQ3_0, GGML_TYPE_TQ3_0V);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TQ3_0, GGML_TYPE_TQ3_0V);
