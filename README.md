# llama.cpp — DualDeadline branch

[![Paper on ResearchGate](https://img.shields.io/badge/paper-ResearchGate-00CCBB)](https://www.researchgate.net/publication/411009097_DualDeadline_Component-Staged_Expert_Prefetching_for_Exact_Offloaded_Mixture-of-Experts_Inference)

> [!NOTE]
> **Looking for TurboQuant?** The 3-bit KV-cache quantization work this
> repository is named after lives on the [`main` branch](../../tree/main).
> This branch (`llama-dualdeadline`) is an independent research prototype.

This branch implements **DualDeadline** — component-staged expert streaming
for offloaded mixture-of-experts inference, after the paper
[*DualDeadline: Component-Staged Expert Prefetching for Exact Offloaded
Mixture-of-Experts Inference*](https://www.researchgate.net/publication/411009097_DualDeadline_Component-Staged_Expert_Prefetching_for_Exact_Offloaded_Mixture-of-Experts_Inference)
(Marcel Butucea, 2026; [study artifacts](https://github.com/unixsysdev/dualdeadline)).

A gated MoE expert has **two transfer deadlines, not one**: gate/up
projections are needed when expert compute starts, but the down projection is
not consumed until gate/up arithmetic finishes. When expert tensors are kept
in pinned host RAM (`--cpu-moe --no-mmap`), this branch lets the GPU claim the
batch-1 expert matmuls by copying only the router-selected experts' slices
into a persistent per-tensor LRU cache — gate/up slices first, down slices
overlapping gate/up compute on a dedicated copy stream. Routing and outputs
are exact; there is no prediction and no approximation.

```sh
GGML_CUDA_DUAL_DEADLINE=1 \
llama-cli -m model.gguf -ngl 999 --cpu-moe --no-mmap -p "..."
```

Measured on Qwen3.6-35B-A3B (Q4_K_XL, 256 experts/layer, top-8) with all
expert tensors in host memory, AMD Ryzen AI MAX+ 395 / Radeon 8060S
(Strix Halo, unified memory), ROCm 7.2.4, `llama-bench -r 3`, tg64:

| configuration | tg64 t/s |
|---|---:|
| upstream: expert matmuls on CPU (16 threads) | 49.7 ± 0.3 |
| **DualDeadline staged**, cache 64 experts/tensor (84% hit rate) | **44.1 ± 0.4** |
| DualDeadline **monolithic**, cache 64 (same copies, single deadline) | 42.7 ± 0.4 |
| DualDeadline staged, cache 32 | 41.6 ± 0.4 |
| DualDeadline monolithic, cache 32 | 39.6 ± 0.4 |

Two honest readings. First, **staged beats monolithic by 3–5% end-to-end** at
identical bytes, cache, and kernels — that is the paper's two-deadline claim
reproduced inside a real decode loop rather than a microbenchmark. Second, on
*this* unified-memory APU the CPU baseline still wins: there is no PCIe link
to hide transfers behind, so explicit staging competes with zero-cost DRAM
reads. The regime this mechanism targets is a PCIe-attached discrete GPU
whose VRAM cannot hold the experts — where the alternative to staging is CPU
compute or full-tensor streaming — and that measurement is still to be done.
A validation mode (`GGML_CUDA_DD_VALIDATE=1`) recomputes every intercepted
matmul via the standard path: **3,120 ops checked, max abs err 0.0** —
bitwise identical.

See [docs/dual-deadline.md](docs/dual-deadline.md) for the design, all
environment variables, limitations, and how this relates to upstream's
used-expert streaming. Implementation:
[`ggml/src/ggml-cuda/dual-deadline.cu`](ggml/src/ggml-cuda/dual-deadline.cu).

---

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp)](https://github.com/ggml-org/llama.cpp/releases)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3A0cc4m%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev branches](https://github.com/ggml-org/llama.cpp-dev/blob/master/README-features.md) / [compile times](https://github.com/ggml-org/llama.cpp-dev/blob/master/README-compile-times.md) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
