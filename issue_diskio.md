# Disk I/O Minimization — Research

Branch: `experiment/diskio-research` (based on `dev` at `62419af`)

## TL;DR

The engine's disk I/O is **already well-engineered on the hottest path** (expert streaming uses coalesced O_DIRECT `pread` + `posix_fadvise` hints + LRU + pin cache + speculative prefetch). There are **4 concrete, bounded opportunities** to shave latency, ranked by ROI:

| # | Opportunity | Where | Frequency | Estimated win |
|---|---|---|---|---|
| 1 | **KV-cache write batching** (157 fwrites/token → 1) | `kv_disk_append` | per turn | cuts ~100s of syscalls/turn |
| 2 | **`/proc/meminfo` fopen storm** | `rss_gb()` | ~every 16 tokens (Linux) | eliminates recurring open/read/close |
| 3 | **Expert prefetch on Windows** (`PrefetchVirtualMemory`) | `expert_prefetch` | per miss | mmap path is Linux/macOS-only today |
| 4 | **KV-cache: buffered handle kept open** | `kv_disk_append` | per turn | kills open+fseek+close per turn |

There are also **2 non-opportunities** worth recording so we don't re-investigate: O_DIRECT for experts (correctly used today), and PagedAttention-style file layout (already single-file + indexed).

---

## How the engine does disk I/O today

There are **three I/O stacks**, behaving very differently:

| Stack | Mechanism | Frequency | Files |
|---|---|---|---|
| **Expert weights** (hottest) | `pread` on kept-open fds + `posix_fadvise`, optional `mmap` | per miss, every token | `st.h`, `glm.c:1328` |
| **KV cache** (`.coli_kv`) | `fopen` + `fwrite`/`fread` | per turn | `glm.c:3812-3889` |
| **Everything else** (config, tokenizer, stats, grammar) | `fopen` + `fread` | startup-only | scattered |

### Expert path (the hot path — already good)

`expert_load` (`glm.c:1328-1481`) has three sub-paths:

- **Default `pread` path** (`glm.c:1385-1472`): coalesces the 3 contiguous expert tensors (gate/up/down) into **one ~19 MB O_DIRECT `pread`** into a 16K-aligned slab (`glm.c:1447`). Falls back to 3 separate `pread`s only if non-contiguous. Scales are 3 tiny separate `pread`s (kilobytes). `posix_fadvise(DONTNEED)` evicts pages after if `g_drop`. **This is well-batched — one syscall for ~19 MB.**
- **`COLI_MMAP=1` path** (`glm.c:1352-1383`): `mmap` per shard fd (cached), `madvise(WILLNEED)` + synchronous page-touch loop. Zero-copy. **Default OFF, and Linux/macOS/FreeBSD-only** — no `MapViewOfFile` on Windows.
- **Prefetch hints**: `expert_prefetch` (`glm.c:1602-1609`) → `st_prefetch` (`st.h:178`) issues `posix_fadvise(WILLNEED)` — readahead hint only, no data read. Called from `moe` next-64-block lookahead, pilot, and SPEC.

### KV cache persistence (per turn — opportunity here)

`kv_disk_append` (`glm.c:3834-3855`), called once per turn:
1. `fopen("r+b")` — **reopens the file every turn**
2. `fseek` to append position
3. **per-position loop**: for each new token, `fwrite` the token i32, then **2 fwrites per layer** (Lc + Rc) + optional DSA Ic. With 78 layers that's **~157 fwrites per token appended**.
4. `fflush` (userspace only — **no fsync/fdatasync anywhere in the codebase**)
5. `fseek` back to header + `fwrite` the new nrec counter (crash-safe ordering)
6. `fclose`

Record size ~182 KB/token. On a long first turn this is **tens of thousands of small fwrites**. stdio buffering coalesces them into fewer `write` syscalls, but the userspace overhead remains.

### Recurring surprise: `/proc/meminfo`

`rss_gb()` (`glm.c:4625`) does `fopen("/proc/meminfo")` + fgets + fclose. Called from every STAT line and every 16-token heartbeat (`glm.c:3473, 3477`). On Linux this is an **open+read+close of procfs ~every 16 tokens**. (Windows uses `compat_meminfo`, no file — not affected.)

---

## What similar projects do

**llama.cpp** (the reference): `mmap`s the entire model read-only, uses `--mlock` to pin hot pages, streams layers to GPU via partial offload, and issues per-pass readahead of upcoming tensors (`llama-mmap.cpp`). Justine Tunney's mmap work: "load 100× faster using half as memory." Crucial finding from discussion #18758: **for MoE, mmap beats O_DIRECT** when the model fits in ~RAM — O_DIRECT takes "at least 10× longer" on repeated loads because it bypasses the page cache that serves re-faults for free.

**The general consensus across llama.cpp, vLLM, AirLLM, PRESERVE, HOBBIT, SolidAttention (FAST '26):**
- mmap + OS page cache as the backing store for an LRU is the proven recipe
- prefetch the *next* expert/layer while computing the current one — this is where the 0.5ms lives
- single indexed file (one `open()`) beats one-file-per-expert
- align tensors to 4KB (preferably 64KB) for clean page-fault boundaries + SSD geometry
- buffer sweet spot ~1MB; syscall cost ~1-5µs each, so batching matters at high repetition

---

## The 4 opportunities (ranked)

### Opportunity 1 — KV-cache write batching (HIGH ROI, LOW risk)

**Problem:** `kv_disk_append` does ~157 `fwrite` calls per appended token (1 token i32 + 2×78 layers). stdio buffering hides some of this, but on a long first-turn prefill (hundreds-thousands of tokens) this is tens of thousands of fwrites.

**Fix:** Build one contiguous record in a heap buffer (token + all layers' Lc/Rc/Ic for that position), then **a single `fwrite` per position** (or even one `fwrite` for the whole turn). The data is already laid out contiguously in memory per-layer (`coli_kv_row`), so a layered `memcpy` into a staging buffer + one write is straightforward.

**Win:** ~157× fewer fwrite calls per token. Even with stdio coalescing, the userspace loop overhead is real at scale.

### Opportunity 2 — `/proc/meminfo` fopen storm (MEDIUM ROI, trivial)

**Problem:** `rss_gb()` opens, reads, closes `/proc/meminfo` every ~16 tokens on Linux. Each is ~3 syscalls + path resolution.

**Fix:** Either (a) cache the value for N tokens (e.g. re-read at most once per second), or (b) keep the fd open and `rewind`+`fgets`. Trivial change.

### Opportunity 3 — Expert prefetch on Windows (MEDIUM ROI, bounded)

**Problem:** The `COLI_MMAP=1` path (which gives zero-copy expert access + free OS-cache re-faults) is **Linux/macOS/FreeBSD-only** — `glm.c:1301` guards it. On Windows, experts always go through the `pread` path, and `expert_prefetch` issues `posix_fadvise(WILLNEED)` which is a no-op shim on Windows (`compat.h`).

**Fix:** On Windows, implement the prefetch via `PrefetchVirtualMemory` (the Win32 analog of `MADV_WILLNEED`) on an mmap'd region, or via an async `ReadFile`+`OVERLAPPED` into a scratch buffer. This brings the Windows build closer to parity with the Linux mmap+prefetch story.

**Scope:** This is the largest of the four — it touches the Windows I/O path. Worth doing if Windows perf is a goal; skip if Linux is the target.

### Opportunity 4 — KV-cache: keep handle open (LOW-MEDIUM ROI, LOW risk)

**Problem:** `kv_disk_append` does `fopen`+...+`fclose` every turn. Handle creation is ~5-15µs of pure overhead (worse on Windows).

**Fix:** Open the KV file once (lazily on first append), keep the `FILE*` for the engine lifetime, just `fseek`+write each turn. Close on shutdown. Pair with Opportunity 1 for the write batching.

---

## Non-opportunities (recording so we don't re-investigate)

- **O_DIRECT for experts**: already correctly used (`st.h:83`, `DIRECT=1`). For an LRU+refetch pattern the page cache is your friend, but the engine offers both paths (O_DIRECT pread default + optional mmap) and the O_DIRECT coalesced read is already one syscall for ~19MB. Don't change this.
- **Single-file layout**: the engine already uses safetensors shards with kept-open fds + offset-indexed tensors (`st.h`). No per-expert open()/close() waste. Don't change this.
- **PagedAttention**: solves concurrency fragmentation this engine doesn't have (≤16 slots). Not applicable.

---

## Next steps

The highest-ROI, lowest-risk starting point is **Opportunity 1 (KV write batching) + Opportunity 4 (keep handle open)** — they're in the same function, both low-risk, and together they eliminate the per-turn open/close overhead and the per-token fwrite storm. Opportunity 2 is a trivial 5-minute fix we can bundle in.

Opportunity 3 (Windows prefetch) is the biggest single win but also the largest scope — separate effort, gated on whether Windows perf is a priority.

## Sources

- [justine.lol/mmap — Edge AI Just Got Faster](https://justine.lol/mmap/)
- [llama.cpp discussion #18758 — Mmap faster than direct I/O for MoE](https://github.com/ggml-org/llama.cpp/discussions/18758)
- [llama.cpp issue #20757 — Two-tier GPU+RAM expert cache](https://github.com/ggml-org/llama.cpp/issues/20757)
- [FAST '26 — Programmable Page Cache for LLM loading](https://www.usenix.org/system/files/fast26-liu-yubo.pdf)
- [FAST '26 — SolidAttention: SSD-based serving](https://www.usenix.org/system/files/fast26-zheng.pdf)
- [HOBBIT — Mixed precision expert offloading](https://arxiv.org/html/2411.01433v2)
- [posix_fadvise(2) — man7.org](https://man7.org/linux/man-pages/man2/posix_fadvise.2.html)
- [madvise(2) — man7.org](https://man7.org/linux/man-pages/man2/madvise.2.html)
- [Microsoft Learn — File Buffering (FILE_FLAG_NO_BUFFERING)](https://learn.microsoft.com/en-us/windows/win32/fileio/file-buffering)
- [Microsoft Learn — PrefetchVirtualMemory](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-prefetchvirtualmemory)
- [What makes system calls expensive — codingconfessions.com](https://blog.codingconfessions.com/p/what-makes-system-calls-expensive)
- [Syscall overhead — Stack Overflow](https://stackoverflow.com/questions/8247331/syscall-overhead)
