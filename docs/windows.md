# Windows 11 (native, no WSL)

colibrì builds and runs natively on Windows 11 x86-64 with MinGW-w64. The port
adds a `_WIN32` compatibility layer in `c/compat.h` that maps POSIX I/O to the
Windows API (pread → ReadFile+OVERLAPPED, posix_fadvise no-op, aligned
allocation, MoveFileEx rename, GlobalMemoryStatusEx RAM detection). All platform
differences stay in `compat.h`; the engine source is unchanged.

**Toolchain:** GCC via [winlibs](https://winlibs.com/) or MSYS2 MinGW-w64.
Tested with GCC 16.1.0 (x86_64-ucrt-posix-seh).

```powershell
# One-time toolchain install (pick one):
scoop install mingw-winlibs                    # portable, no shell needed
# or: pacman -S mingw-w64-x86_64-gcc make     # via MSYS2

# Build (from c/ directory):
make glm.exe            # GLM-5.2 engine (static, no DLL dependencies)
make olmoe.exe          # OLMoE engine (same shims)
make iobench.exe        # disk I/O benchmark
make test-c             # run C tests
make test-python        # run Python tests (requires python)

# AVX-VNNI: Intel Alder Lake+ (and Meteor Lake+) CPUs have a 128-bit int8
# dot-product instruction (VPDPBUSD) the engine can use for ~1.3x faster
# quantized matmul. The x86-64-v3 default (portable AVX2) compiles it out;
# build for THIS machine to enable it:
make glm.exe ARCH=native                       # banner prints "idot: avx-vnni"

# Verify (tiny model, 2.4 MB):
pip install torch transformers safetensors huggingface_hub
python tools/make_glm_oracle.py                # generate tiny oracle
SNAP=./glm_tiny TF=1 ./glm.exe 64 16 16        # expect "32/32 positions"

# Run with real model:
SNAP=D:\glm52_i4 ./glm.exe 64 4 16            # batch inference
python coli chat --model D:\glm52_i4            # interactive chat
python coli serve --model D:\glm52_i4            # OpenAI-compatible API
```

> Windows Store's `python` alias stub is the single most common native-Windows
> trap: install real Python (python.org or `winget install Python.Python.3.12`)
> or disable the alias under *Settings → Apps → App execution aliases*.

## Warmup (overnight cache priming)

The engine's expert cache learns from your workload. The included `warmup.ps1`
script runs `coli run` in a loop with diverse prompts to build the
`.coli_usage` histogram unattended, so the next real session starts with a
large, accurate hot-expert pin. Each run saves usage atomically on clean
completion.

```powershell
.\warmup.ps1 -Rounds 1 -Ngen 32               # ~60-90 min, durable progress
```

## NVIDIA GPU (optional, via runtime DLL)

On Windows the engine is built with MinGW gcc but CUDA kernels require MSVC +
nvcc. The split is clean: build the CUDA backend into a standalone
`coli_cuda.dll` (nvcc + MSVC), then the host `glm.exe` loads it at runtime via
`LoadLibrary` (`c/backend_loader.c`). The host never links cudart directly; if
the DLL is absent the engine falls back to CPU without error.

```powershell
# Prerequisites: CUDA Toolkit + MSVC Build Tools (cl.exe) + nvcc on PATH.
# Build the DLL from a shell with the MSVC environment set (vcvars64.bat or
# "x64 Native Tools Command Prompt for VS"):
make cuda-dll CUDA_HOME="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8" CUDA_ARCH=sm_120

# Build the host with the runtime loader (CUDA_DLL=1 adds -DCOLI_CUDA and
# links backend_loader.o instead of cudart):
make glm.exe CUDA_DLL=1 ARCH=native

# Run with the GPU expert tier (8 GB VRAM budget here; scale to your free VRAM):
$env:COLI_CUDA="1"; $env:COLI_GPU="0"; $env:CUDA_EXPERT_GB="8"
python coli chat --model D:\glm52_i4 --topp 0.7
```

The DLL exports the full `extern "C"` surface (including the #111 pipeline ABI);
`backend_loader.c` resolves symbols via `GetProcAddress` on first use.
`ColiCudaTensor*` is opaque to the host (stored, never dereferenced), so the
MSVC-allocated struct is safe across the ABI boundary. `CUDA_ARCH` must match
your GPU's compute capability (e.g. `sm_120` for Blackwell / RTX 50-series,
`sm_89` for Ada / RTX 40-series). A one-shot `build_cuda.bat` wrapper is also
available.

**Measured on a single RTX 5070 Ti + Core Ultra 9 (32 GB RAM):** CPU-only 0.63
→ CUDA attention+dense 0.72 → **1.07 tok/s** with the GPU-resident pipeline at
decode ([#273](https://github.com/JustVugg/colibri/issues/273), merged in #274).
