# --- OS detection ---
# On Windows the OS env var is 'Windows_NT' in EVERY shell (cmd, PowerShell,
# MSYS2, Git-Bash), and GNU Make imports it. Detect Windows from it FIRST so
# `make` works from a native PowerShell/CMD shell — there `uname` is not on
# PATH, so the old uname-only check returned empty and fell through to the
# Linux branch (no .exe, no -static). We only call `uname` when NOT on Windows.
ifeq ($(OS),Windows_NT)
IS_WIN   := 1
UNAME_S  :=
UNAME_M  :=
else
IS_WIN   :=
UNAME_S  := $(shell uname -s)
UNAME_M  := $(shell uname -m)
endif

ifeq ($(UNAME_S),Darwin)
# --- macOS / Apple Silicon ---
# Apple clang non include il runtime OpenMP: se c'e' libomp di Homebrew lo usa
# (brew install libomp), altrimenti compila single-thread (i pragma omp sono ignorati).
# Niente -march: su arm64 NEON e' baseline (i kernel __ARM_NEON si attivano da soli).
CC      = clang
OMPDIR := $(shell brew --prefix libomp 2>/dev/null)
ifneq ($(OMPDIR),)
OMPC    = -Xclang -fopenmp -I$(OMPDIR)/include
OMPL    = -L$(OMPDIR)/lib -lomp
else
$(warning libomp not found: building single-threaded. For multithreading: brew install libomp)
OMPC    =
OMPL    =
endif
CFLAGS  = -O3 $(OMPC) -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm $(OMPL)
EXE     =
else ifneq ($(IS_WIN),)
# --- Windows 11 x86-64 (MinGW-w64 / MSYS2) ---
# GCC + libgomp + winpthreads: pthread, OpenMP, clock_gettime, opendir/readdir,
# AVX2 intrinsics - tutto gratis, nessun porting.
# ARCH default = x86-64-v3 (portable binary with AVX2). For max speed on THIS
# machine use ARCH=native: on AVX-VNNI CPUs (Intel Alder Lake+, Meteor Lake+)
# it also unlocks the 128-bit VPDPBUSD int8/int4 dot kernel (dot_i8i8/dot_i4i8),
# which the x86-64-v3 baseline does not define. The #ifdef guards in glm.c mean
# a v3 build simply compiles out the VNNI path - safe on any x86-64.
CC      = gcc
ARCH   ?= x86-64-v3
CFLAGS  = -D_FILE_OFFSET_BITS=64 -O3 -march=$(ARCH) -fopenmp -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm -fopenmp -static
EXE     = .exe
else
# --- Linux x86-64 (percorso originale, invariato) ---
CC      = gcc
# ARCH=native -> ottimizzato per QUESTA macchina (default, piu' veloce).
# ARCH=x86-64-v3 -> binario PORTABILE su qualsiasi x86-64 moderno con AVX2 (per distribuire).
# ARCH=x86-64 -> massima compatibilita' (niente AVX2: usa il path scalare di fallback).
ARCH   ?= native
CFLAGS  = -O3 -march=$(ARCH) -fopenmp -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm -fopenmp
EXE     =
endif

# Optional io_uring async expert reads (PIPE=2). Set IOURING_PREFIX to a liburing
# install prefix, or IOURING=1 to auto-probe /usr and /tmp/liburing-install.
IOURING       ?= 0
IOURING_PREFIX ?=
IOURING_CFLAGS  =
IOURING_LDFLAGS =
ifeq ($(IOURING),1)
ifneq ($(IOURING_PREFIX),)
IOURING_CFLAGS  = -DCOLI_IOURING -I$(IOURING_PREFIX)/include
IOURING_LDFLAGS = -L$(IOURING_PREFIX)/lib -luring
else ifneq ($(wildcard /tmp/liburing-install/include/liburing.h),)
IOURING_CFLAGS  = -DCOLI_IOURING -I/tmp/liburing-install/include
IOURING_LDFLAGS = -L/tmp/liburing-install/lib -Wl,-rpath,/tmp/liburing-install/lib -luring
else ifneq ($(wildcard /usr/include/liburing.h),)
IOURING_CFLAGS  = -DCOLI_IOURING
IOURING_LDFLAGS = -luring
endif
endif

# CUDA=1 adds an opt-in backend for resident tensors. ROCM=1 selects the same
# backend ABI with HIP/ROCm. The default build remains pure C and keeps the
# original zero-dependency runtime.
#
# Two paths:
#   - Linux/macOS: CUDA=1 links backend_cuda.o directly (cudart via -l).
#   - Linux/WSL:   ROCM=1 compiles backend_cuda.cu with hipcc and links
#                  amdhip64; the source uses a small HIP compatibility layer.
#   - Windows:     ROCM=1 uses MinGW gcc for C host objects and the HIP SDK's
#                  hipcc for device compilation/linking (set ROCM_HOME or put
#                  hipcc on PATH).
#   - Windows:      CUDA_DLL=1 builds a standalone coli_cuda.dll (nvcc+MSVC),
#                   then the host glm.exe loads it at runtime via backend_loader.c
#                   (LoadLibrary/GetProcAddress). MinGW gcc cannot compile .cu
#                   (nvcc needs cl.exe), and cross-linking MSVC objects into a
#                   gcc binary is fragile — the DLL split keeps the toolchains
#                   clean. See backend_loader.c and README "cuda-dll" below.
CUDA      ?= 0
CUDA_DLL  ?= 0
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= native
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH) -Xcompiler=-Wall,-Wextra
ROCM      ?= 0
ROCM_HOME ?= $(if $(HIP_PATH),$(HIP_PATH),/opt/rocm)
HIPCC     ?= $(if $(IS_WIN),hipcc,$(ROCM_HOME)/bin/hipcc)
ROCM_ARCH ?= gfx1101
HIPFLAGS  ?= -O3 -std=c++17 -fPIC --offload-arch=$(ROCM_ARCH) -I"$(ROCM_HOME)/include"
PYTHON    ?= python3
CUDA_OBJ  =
ROCM_OBJ  =
TEST_BINS = tests/test_json$(EXE) tests/test_st$(EXE) tests/test_tier$(EXE) tests/test_grammar$(EXE) tests/test_decode_batch$(EXE) tests/test_idot$(EXE) tests/test_i4_acc512$(EXE)

# Windows CUDA DLL path: host links the loader, NOT cudart.
ifneq ($(IS_WIN),)
ifeq ($(CUDA_DLL),1)
CFLAGS  += -DCOLI_CUDA
CUDA_OBJ = backend_loader.o
endif
endif

# Linux CUDA direct-link path (unchanged).
ifeq ($(CUDA),1)
ifeq ($(UNAME_S),Darwin)
$(error CUDA=1 is supported only on Linux)
endif
ifneq ($(IS_WIN),)
# On Windows use CUDA_DLL=1 (runtime DLL), not CUDA=1 (direct link).
$(error On Windows use: make CUDA_DLL=1 cuda-dll  (see backend_loader.c))
endif
CFLAGS   += -DCOLI_CUDA
LDFLAGS  += -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart -lstdc++
CUDA_OBJ  = backend_cuda.o
endif

# Linux/WSL ROCm path. Keep the existing COLI_CUDA symbols and CLI variables
# as the stable accelerator ABI; COLI_ROCM only selects HIP compilation and
# disables the NVIDIA-only WMMA fast path in backend_cuda.cu.
ifeq ($(ROCM),1)
ifneq ($(CUDA),0)
$(error CUDA=1 and ROCM=1 are mutually exclusive)
endif
ifneq ($(CUDA_DLL),0)
$(error CUDA_DLL=1 and ROCM=1 are mutually exclusive)
endif
ifeq ($(UNAME_S),Darwin)
$(error ROCM=1 is supported on Linux/WSL, not macOS)
endif
CFLAGS     += -DCOLI_CUDA -DCOLI_ROCM
ROCM_OBJ   = backend_hip.o
ifeq ($(IS_WIN),)
LDFLAGS    += -L$(ROCM_HOME)/lib -Wl,-rpath,$(ROCM_HOME)/lib -lamdhip64 -lstdc++
else
# The Windows HIP SDK owns the clang/clang++ link step and supplies the
# runtime import library. Keep MinGW gcc for the C host objects because the
# HIP SDK alone does not ship pthread.h; hipcc links those objects with HIP.
ROCM_LINK  = "$(HIPCC)"
CFLAGS     := -D_FILE_OFFSET_BITS=64 -O3 -march=$(ARCH) -fopenmp -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function -DCOLI_CUDA -DCOLI_ROCM
LDFLAGS    := -fopenmp
endif
endif

GPU_OBJ = $(CUDA_OBJ) $(ROCM_OBJ)
ROCM_LINK ?= $(CC)
GLM_INPUT = glm.c
HY3_INPUT = hy3.c

# Native Windows ROCm uses MinGW for the C host translation unit and hipcc for
# the final link. This keeps pthread/OpenMP headers available while still
# linking amdhip64 and the HIP device object.
ifeq ($(ROCM),1)
ifeq ($(IS_WIN),1)
GLM_INPUT = glm_host.o
HY3_INPUT = hy3_host.o

glm_host.o: glm.c st.h json.h tok.h tok_unicode.h compat.h grammar.h
	$(CC) $(CFLAGS) -c $< -o $@

hy3_host.o: hy3.c st.h json.h compat.h
	$(CC) $(CFLAGS) $(IOURING_CFLAGS) -c $< -o $@
endif
endif

# METAL=1 adds an opt-in Apple-GPU backend (macOS only). The shader is compiled at
# runtime, so no Xcode / offline metal compiler is required. Default build unchanged.
METAL     ?= 0
METAL_OBJ  =
METALXX    = clang++ -x objective-c++ -fobjc-arc -O3
ifeq ($(METAL),1)
ifneq ($(UNAME_S),Darwin)
$(error METAL=1 is supported only on macOS)
endif
CFLAGS   += -DCOLI_METAL
LDFLAGS  += -framework Metal -framework Foundation -lc++
METAL_OBJ = backend_metal.o
endif

all: glm$(EXE) hy3$(EXE)

# phony 'glm' → 'glm.exe' on Windows (so 'make glm' and 'coli build' work on every platform)
ifneq ($(EXE),)
glm: glm$(EXE)
hy3: hy3$(EXE)
endif

glm$(EXE): $(GLM_INPUT) st.h json.h tok.h tok_unicode.h compat.h grammar.h $(GPU_OBJ) $(METAL_OBJ)
	$(ROCM_LINK) $(CFLAGS) $(GLM_INPUT) $(GPU_OBJ) $(METAL_OBJ) -o glm$(EXE) $(LDFLAGS)

# Windows runtime loader object: resolves coli_cuda_* from coli_cuda.dll.
backend_loader.o: backend_loader.c backend_cuda.h compat.h
	$(CC) $(CFLAGS) -c backend_loader.c -o $@

# Windows CUDA DLL: compile backend_cuda.cu with nvcc (+MSVC cl.exe as host
# compiler, required by nvcc on Windows) into coli_cuda.dll. Run this from a
# shell that has the MSVC environment set (e.g. after vcvars64.bat, or from a
# "x64 Native Tools Command Prompt"). COLI_CUDA_BUILDING_DLL enables
# __declspec(dllexport) so the 11 API symbols are exported.
cuda-dll: backend_cuda.cu backend_cuda.h
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	@command -v cl >/dev/null 2>&1 || { echo "cl.exe (MSVC) not in PATH — run vcvars64.bat first" >&2; exit 1; }
	$(NVCC) $(NVCCFLAGS) -shared -DCOLI_CUDA_BUILDING_DLL \
		-L"$(CUDA_HOME)/lib/x64" -lcudart \
		backend_cuda.cu -o coli_cuda.dll

backend_cuda.o: backend_cuda.cu backend_cuda.h
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	$(NVCC) $(NVCCFLAGS) -c backend_cuda.cu -o $@

# ROCm/HIP build. backend_cuda.cu is intentionally shared with CUDA so the
# numerical implementation stays covered by the same backend test suite.
backend_hip.o: backend_cuda.cu backend_cuda.h
	@command -v "$(HIPCC)" >/dev/null 2>&1 || { echo "hipcc not found: set ROCM_HOME or HIPCC" >&2; exit 1; }
	"$(HIPCC)" $(HIPFLAGS) -DCOLI_CUDA -DCOLI_ROCM -c backend_cuda.cu -o $@

backend_metal.o: backend_metal.mm backend_metal.h
	$(METALXX) -c backend_metal.mm -o $@

metal-test: tests/test_backend_metal.mm backend_metal.mm backend_metal.h
	$(METALXX) tests/test_backend_metal.mm backend_metal.mm -framework Metal -framework Foundation -o backend_metal_test
	./backend_metal_test

cuda-test: backend_cuda.cu backend_cuda.h tests/test_backend_cuda.cu
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	$(NVCC) $(NVCCFLAGS) backend_cuda.cu tests/test_backend_cuda.cu -o backend_cuda_test$(EXE)
	./backend_cuda_test$(EXE)

hy3$(EXE): $(HY3_INPUT) st.h json.h compat.h $(GPU_OBJ)
	$(ROCM_LINK) $(CFLAGS) $(IOURING_CFLAGS) $(HY3_INPUT) $(GPU_OBJ) -o hy3$(EXE) $(LDFLAGS) $(IOURING_LDFLAGS)

rocm-test: backend_cuda.cu backend_cuda.h tests/test_backend_cuda.cu
	@command -v "$(HIPCC)" >/dev/null 2>&1 || { echo "hipcc not found: set ROCM_HOME or HIPCC" >&2; exit 1; }
	"$(HIPCC)" $(HIPFLAGS) -DCOLI_CUDA -DCOLI_ROCM backend_cuda.cu tests/test_backend_cuda.cu -o backend_hip_test$(EXE)
	./backend_hip_test$(EXE)

cuda-bench: backend_cuda.cu backend_cuda.h tests/bench_tensor_core.cu
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "nvcc not found: set CUDA_HOME or NVCC" >&2; exit 1; }
	$(NVCC) $(NVCCFLAGS) backend_cuda.cu tests/bench_tensor_core.cu -o backend_cuda_bench$(EXE)
	./backend_cuda_bench$(EXE)

olmoe$(EXE): olmoe.c st.h json.h compat.h
	$(CC) $(CFLAGS) olmoe.c -o olmoe$(EXE) $(LDFLAGS)

# binario portabile da distribuire su altre macchine x86-64
portable:
	$(MAKE) glm$(EXE) ARCH=x86-64-v3

iobench$(EXE): iobench.c compat.h
	$(CC) $(CFLAGS) iobench.c -o iobench$(EXE) $(LDFLAGS)

tests/test_json$(EXE): tests/test_json.c json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_st$(EXE): tests/test_st.c st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_tier$(EXE): tests/test_tier.c tier.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_grammar$(EXE): tests/test_grammar.c grammar.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_decode_batch$(EXE): tests/test_decode_batch.c decode_batch.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_idot$(EXE): tests/test_idot.c glm.c st.h json.h tok.h tok_unicode.h compat.h grammar.h tier.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_i4_acc512$(EXE): tests/test_i4_acc512.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-c: $(TEST_BINS)
	@for test in $(TEST_BINS); do ./$$test || exit 1; done

test-python:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

test: test-c test-python

# Local validation: one portable CPU build and dependency-free tests.
check:
	$(MAKE) clean
	$(MAKE) portable
	$(MAKE) test

clean:
	rm -f olmoe$(EXE) hy3$(EXE) glm$(EXE) iobench$(EXE) glm_host.o hy3_host.o backend_cuda.o backend_hip.o backend_loader.o backend_cuda_test$(EXE) backend_hip_test$(EXE) backend_cuda_bench$(EXE) backend_hip_bench$(EXE) backend_metal.o backend_metal_test coli_cuda.dll coli_cuda.lib coli_cuda.exp $(TEST_BINS)
	rm -rf tests/__pycache__

.PHONY: all glm hy3 cuda-test rocm-test cuda-bench cuda-dll portable test-c test-python test check clean
