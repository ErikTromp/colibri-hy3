# Frozen-corpus speculative drafts

*The canonical reference for the `COLI_DRAFT_CORPUS=` draft source. It is the
existing n-gram draft (method E) taken off the live context and pointed at a
persistent file of previously generated token ids — the retrieval-based
speculative decoding idea, reusing this engine's verification unchanged.*

## The mechanism in one paragraph

`ngram_draft()` already proposes a continuation by matching the last two tokens
of the **current** sequence against earlier positions in that same sequence. A
frozen corpus generalizes that: the engine loads a file of token ids produced by
past runs, and at each decode step proposes the continuation that followed the
**longest suffix of the live context** found anywhere in that corpus (suffix
lengths 8 down to 3, most recent occurrence first). The proposal enters the same
batch-union verify forward as MTP, n-gram and grammar drafts, so it is only ever
a *proposal*: the target model's own argmax decides what is accepted, and the
output is what the engine would have produced without any drafting. Unlike MTP
(one token per forward) a corpus hit proposes a whole span at once.

## What it costs and what it pays

Measured on GLM-5.2 int4, 96-token greedy decode, replaying a prompt whose
generation is in the corpus:

| Host | Baseline (MTP) | With corpus drafts |
|---|---|---|
| H200, fully resident | 1.78 tok/forward, 3.69 tok/s | 6.00 tok/forward, **4.52 tok/s** (+22%), 90% acceptance |
| CPU (24C Xeon, 120 GB pin) | 1.76 tok/forward, 0.82 tok/s | 7.92 tok/forward, **1.00 tok/s** (+22%), 100% acceptance |

Note the gap between the two multipliers: **3–4× fewer forwards buys ~22% of
wall clock, not 3–4×**. In an MoE every row of a verify batch activates its own
experts, so speculation amortizes the dense path, attention and per-forward
overhead, but never the expert work itself — that scales with rows. The same
ceiling showed up on a disk-streaming host, where the extra rows also widen the
per-forward expert union and cost LRU hit rate. Treat the forward count as the
thing being optimized and measure wall clock separately.

## When this helps, and when it does not

Read the two numbers above as a **best case, not a general speed-up**: they come
from replaying a prompt whose generation is already in the corpus, so nearly
every step has a long suffix to match. That is the shape of a benchmark replay,
a regression harness, or a workload that answers the same questions repeatedly.

On genuinely novel text the corpus has nothing to propose, drafts fall back to
MTP/n-gram, and the gain approaches zero (the lookup itself is cheap, but it is
not free). A general chat assistant answering new questions is the case where
this buys the least. The feature is opt-in for exactly that reason — unset
`COLI_DRAFT_CORPUS` and nothing about decoding changes.

Related: deep drafts from this source are what surfaced
[#689](https://github.com/JustVugg/colibri/issues/689) — speculative verify
batches at `S>=8` diverging from the unbatched path by a near-tie token on CUDA.
That is a separate, open bug in the verify path rather than in this draft
source, but a large `COLI_DRAFT_CORPUS` hit rate is the easiest way to reach it.

Drafts that are *rejected* cost real time, so the source pauses itself below
`COLI_CORPUS_MINACC` acceptance (default 50%, the measured break-even: 90%
acceptance gave +22%, 19% gave −25%). A prompt with no relation to the corpus
costs nothing at all — the suffix match simply finds nothing and the source
stays inert (measured: identical forward count to a run with no corpus). The
risk lives in the *neighbourhood* of the corpus, where near-miss prefixes
produce spurious matches; that is what the guard is for.

## Usage

```bash
# 1. freeze a run's output ids (TOKENS=1 already dumps them to stderr)
TOKENS=1 COLI_TEMP=0 ./coli run --ngen 96 "..." 2>&1 |
  sed -n 's/.*\[TOKENS\][0-9 ]*generated://p' > corpus.txt

# 2. draft from it on later runs
COLI_DRAFT_CORPUS=corpus.txt COLI_CORPUS_K=8 COLI_TEMP=0 ./coli run --ngen 96 "..."
```

- `COLI_DRAFT_CORPUS=<file>` — whitespace-separated token ids. `-1` separates
  spans; a proposal never crosses one. Absent or unreadable = source off.
- `COLI_CORPUS_K=n` — proposal depth (default 8, capped at 48 by the verify
  batch). Deeper proposals raise the forward multiplier and the per-forward
  cost; 8 was the best measured trade on GLM-5.2.
- `COLI_CORPUS_MINACC=pct` — pause threshold (default 50). Below this acceptance
  over a 24-proposal window the source pauses for 256 tokens, then re-arms.
- The engine reports at end of run:
  `corpus drafts: NN% acceptance (a/b proposed from N frozen ids)`.

The source is **off by default** and takes priority over MTP when it has a
proposal; MTP fills the gaps where the corpus has no match.

## Losslessness

The corpus never constrains sampling — it only proposes, and the existing verify
loop accepts a token solely when it matches what the model itself produced. Two
independent checks:

- the token-exact oracle passes with the source **armed** (`SNAP=./glm_tiny
  COLI_DRAFT_CORPUS=... ./colibri 64 16 16` → 20/20, same as unarmed);
- a 96-token CPU generation with 100% acceptance at depth 8 is **byte-identical**
  to the same generation with the source off.

One caveat, and it is not specific to this source: on the CUDA path a run whose
drafts are accepted in long streaks can diverge from the unbatched path by a
single near-tie token (measured: 1 token in 96, at position 85, first 84
identical). Deep verify batches are not bit-identical to S=1 forwards there, and
the same effect shows up as a lower acceptance rate on GPU (90%) than on CPU
(100%) for the identical corpus. The CPU path is byte-exact.
