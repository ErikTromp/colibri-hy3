"""fmt=8 repack tool: Z.ai GLM-5.2-FP8 shards -> the engine's native-FP8-passthrough
container (byte-preserved, no dequant/requant).

fmt=8 is a PUBLIC ordinal. This tool's output format was minted fmt=6 during
this branch's original development, before dev's own #465 (E8/IQ3) claimed
that ordinal upstream and merged it into dev as a REAL fmt=6 (colibri.c's
qt_resolve_fmt, quant.h's E8 constants); re-tagged fmt=100 (PRIVATE ORDINAL
BLOCK, see colibri.c's QT struct comment) from that point forward -- this
tool has never emitted a container claiming ordinal 6 -- graduated to fmt=7
when the maintainer assigned that ordinal on #524, and renumbered to fmt=8
after #705 merged MXFP4 as fmt=7 while this PR was open. Nothing this tool
writes carries the ordinal, so neither renumber changed its output bytes.

Unlike convert_fp8_to_int4.py (which DEQUANTIZES fp8 -> f32 -> REQUANTIZES to a
different, lossy format), this tool copies the fp8 weight bytes AS-IS and only
renames/reshapes the scale sidecar to the engine's `.qs` convention. Same 8
bits/weight streaming cost as the source, zero additional quantization loss.

THE DESIGN LANDMINE (see qt_resolve_fmt in colibri.c, c/quant.h's E4M3_LUT
comment): the engine tells fmt=8 (this tool's output) apart from fmt=1 (plain
int8) ONLY by scale-array geometry -- per-row for fmt=1, per-128x128-block for
fmt=8. This tool must therefore emit .qs sidecars whose byte count is EXACTLY
ceil(O/128)*ceil(I/128)*4, never anything that could coincide with O*4 by
accident for the shapes it targets -- ENFORCED, not just asserted, by
_check_geometry below (maintainer review, #528): it refuses to repack any
[O,I] where nblkO*nblkI==O, the exact shape where this tool's own block-scale
byte count would coincide with a per-row int8 scale byte count. This is not
hypothetical: GLM-5.2's own self_attn.o_proj.weight ([6144,16384]) is a real
instance (nblkO=48, nblkI=128, product=6144==O). It matters more now than it
used to, because the engine's UNSTAMPED read-side collision policy resolves an
ambiguous shape to fmt=1 rather than refusing (see qt_resolve_fmt's
INVERSION) -- so an fmt=8 container this tool emitted at such a shape would be
silently read back as plain int8, not caught by a read-side refusal. Avoiding
the collision at write time is therefore load-bearing, not a redundant
belt-and-braces check. At I=98 specifically, a single-block fp8 tensor's raw
weight-byte count ALSO coincides with a genuine fmt=6 (E8/IQ3) tensor's -- see
qt_resolve_fmt's "SECOND DESIGN LANDMINE" comment; this tool does not target
I=98 shapes in practice (Z.ai's real projections don't land there), and that
narrower collision is unchanged by this PR (still an unconditional read-side
refusal), so _check_geometry does not additionally guard against it here.

SCALE ENCODING: this tool always emits f32 block scales (4 bytes/block, the
`.to(torch.float32)` below) -- the ENCODING this build implements. It never
reads or emits a UE8M0 (1 byte/block) sidecar: Z.ai's GLM-5.2-FP8 checkpoints
ship f32 `weight_scale_inv`, so there is no source data this tool would need
to convert from. A DeepSeek-V4-shaped source (same weight geometry, UE8M0
scales) is out of scope for this tool as written; the read side (qt_resolve_fmt)
recognizes and refuses that byte signature by name rather than misread it,
should a container carrying it ever reach the engine some other way.

SELECTED kinds only: "resident" tensors (shared expert, o_proj, other
attention projections, dense-MLP first layers, and the generic resident
fallback) -- routed experts (kind "x" in convert_fp8_to_int4.classify) are
EXCLUDED and stay on the existing int4-g64 streaming path; routers/norms/bias
(kind "f32") and embed/lm_head (kind "io", BF16 in source, never FP8) are
untouched by this tool -- carrying those into a mixed fmt=8+int4 loadout
directory is separate, deferred "loadout index-rewrite blending" work.

kv_b_proj (kind "kvb") is ALSO deliberately excluded, despite being a resident
tensor classify() would otherwise select: colibri.c's MLA-absorption CPU path
(qt_addrow/qt_matvec_rows, called only on l->kv_b -- the always-available
fallback whenever the Metal fused decode kernel isn't used) has no fmt==8
support. FIX ROUND 2 update: an fmt=8 (or fmt=6) tensor reaching either
function now refuses loudly (exit(1), naming the function and the fmt) --
before that fix it was worse than "misread as int2": qt_addrow SIGSEGV'd
outright (t->q4 is NULL for fmt=8; the untouched int2 fall-through
dereferenced it), and only fmt=6 (t->q4 non-NULL, wrong per-row scale
geometry) actually produced silently-wrong values. Either way, this tool
still does not select kv_b_proj: refusing loudly at the absorb call site is
not the same as SUPPORTING fmt=8 there (no dequant path exists, whether it
now trips a controlled refusal or, before the fix, undefined behavior). The
CUDA absorb kernels (coli_cuda_attention_absorb/_kvdev in backend_cuda.cu)
are similarly int4-specific and unaffected by that fix. Repacking kv_b_proj
to fmt=8 needs real fmt=8 CPU+CUDA absorb-path support first -- out of scope
for this build, so this tool refuses to produce a container the engine
cannot safely read.

METADATA STAMP (reference implementation of the FORMATS-registry FR -- see
docs/FORMATS.md): every output shard's safetensors `__metadata__` carries a
`colibri.fmt` key whose VALUE is itself a JSON-encoded object mapping each
stamped tensor's exact name to its format NAME string (FORMAT_NAME below,
"fp8-e4m3-b128" -- never the internal fmt=8 ordinal, which the container must
never depend on, see colibri.c's QT struct comment). The reader
(qt_from_disk's qt_verify_fmt_stamp, colibri.c) is TRUST-VERIFY-REFUSE: a
stamp that agrees with the byte-arithmetic inference is a no-op (the tensor
loads exactly as it would unstamped); a stamp that disagrees, or names a
format this build doesn't recognize, is refused loudly (same "untrusted
container" discipline as qt_resolve_fmt's own THE DESIGN LANDMINE refusal).
What a stamp additionally lets resolve differs by collision (FIX ROUND 2:
corrected a stale pre-INVERSION description here -- see qt_resolve_fmt's own
documentation for the exact rule in both cases): for the fmt=6-vs-fmt=8
collision, an unstamped tensor at that shape still refuses unconditionally --
a stamp naming the correct candidate resolves it instead, as originally
designed. For the fmt=1-vs-fmt=8 collision, the INVERSION (maintainer review,
#528) means an unstamped tensor at that shape no longer refuses at all -- it
resolves to int8-row by default -- so a stamp's role there is letting a
genuinely-stamped fmt=8 tensor still be read as fmt=8 (overriding that
default), not resolving a refusal that no longer happens. Unstamped
containers (any container from a tool that doesn't stamp, including every
container this tool itself has ever produced before this feature existed)
are otherwise unaffected: no stamp means byte-arithmetic inference alone
decides, exactly as it always has.

HARD CONSTRAINT for this build: unit-tested on synthetic fixtures ONLY (see
tests/test_fp8_repack.py, built with tools/glm_fp8_emit.py's exact real-
checkpoint layout) plus --dry-run inventory mode against those same fixtures.
NO read of any real Z.ai shard, NO full or partial repack of a real checkpoint
-- that is later, user-GO'd work once this build's plumbing has been reviewed.

Usage (synthetic fixtures / local testing only):
  python3 tools/repack_fp8_passthrough.py --indir <fp8_dir> --outdir <out> --dry-run
  python3 tools/repack_fp8_passthrough.py --indir <fp8_dir> --outdir <out> --n-layers 78
"""
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from convert_fp8_to_int4 import classify, check_or_record_params  # reuse: same taxonomy,
                                                                    # same #383-class params guard

# Kinds classify() can return that this tool targets: resident weights, NOT routed
# experts ("x"), NOT the always-F32 set ("f32"), NOT embed/lm_head ("io" -- BF16 in
# source, never FP8), NOT sidecars/skips ("consumed"/"skip"). "kvb" (kv_b_proj) is
# ALSO excluded -- see the module docstring: the CPU absorb path (qt_addrow/
# qt_matvec_rows) and the CUDA absorb kernels have no fmt=8 case yet.
RESIDENT_KINDS = frozenset({"sh", "o", "attn", "dmlp", "q"})

# The format's PUBLIC identity (what containers/FRs advertise) -- the internal
# fmt=8 ordinal is a colibri.c-only enum value, never itself persisted (see
# that file's QT struct comment). Must match the reader's FMT_NAMES table
# (colibri.c, near qt_resolve_fmt) or every stamp this tool writes will be
# refused as "unrecognized format name" on load.
FORMAT_NAME = "fp8-e4m3-b128"

# safetensors __metadata__ key this tool stamps: JSON-encoded {tensor_name:
# format_name}. Kept as a module constant so the reader-side doc and any
# future tooling can cite the exact string instead of re-typing it.
METADATA_KEY = "colibri.fmt"

BLOCK = 128


def _nblk(n):
    return (n + BLOCK - 1) // BLOCK


def is_repack_target(name, dtype, keys, n_layers):
    """True if `name` is a resident-kind FP8 tensor this tool should byte-preserve
    into the fmt=8 container. `keys` is the full set of tensor names in the shard
    (needed to confirm the `_scale_inv` sidecar is actually present -- classify()
    alone can't tell FP8 tensors from any other dtype a resident-kind name might
    carry in a non-FP8 checkpoint variant)."""
    if name.endswith("_scale_inv"):
        return False                       # sidecar, handled together with its weight
    kind = classify(name, n_layers)
    if kind not in RESIDENT_KINDS:
        return False
    if dtype not in ("F8_E4M3", "float8_e4m3fn"):
        return False
    return (name + "_scale_inv") in keys


def _check_geometry(name, O, I, nblkO, nblkI):
    """Refuse (loud, ValueError) rather than silently emit a .qs whose geometry
    doesn't match what qt_resolve_fmt expects for [O,I] -- the same "untrusted
    container" discipline the C side applies on read, applied here on write so a
    malformed source checkpoint is caught at repack time, not at load time three
    steps later with a confusing engine-side refusal.

    WRITER-SIDE REFUSAL (maintainer review, #528): also refuses at the AMBIGUOUS
    shape THE DESIGN LANDMINE describes (qt_resolve_fmt in colibri.c) -- where
    O == ceil(O/128)*ceil(I/128), i.e. this tensor's block-scale byte count
    (nblkO*nblkI*4) exactly equals a per-row int8 scale byte count (O*4). GLM-5.2's
    own self_attn.o_proj.weight ([6144,16384]) is a REAL, non-hypothetical instance
    of this shape. The engine's reader now resolves an unstamped collision to fmt=1
    (the incumbent, decodable format) rather than refusing -- which makes it load
    bearing that THIS tool never emits an fmt=8 container at such a shape: this
    docstring's own opening promise ("never anything that could coincide with O*4
    by accident") is enforced here, not just asserted."""
    exp_nblkO, exp_nblkI = _nblk(O), _nblk(I)
    if (nblkO, nblkI) != (exp_nblkO, exp_nblkI):
        raise ValueError(
            f"{name}: scale_inv block shape ({nblkO},{nblkI}) != expected "
            f"({exp_nblkO},{exp_nblkI}) for [{O},{I}] -- refusing to repack a shape "
            f"the engine's qt_resolve_fmt would either refuse or (worse) misread")
    if nblkO * nblkI == O:
        raise ValueError(
            f"{name}: [{O},{I}] is an AMBIGUOUS fp8-e4m3-b128 shape -- its "
            f"block-scale byte count (nblkO*nblkI*4={nblkO*nblkI*4}) exactly "
            f"coincides with a per-row int8 scale byte count (O*4={O*4}) for the "
            f"same [O,I]; refusing to emit an fmt=8 container the engine's "
            f"qt_resolve_fmt collision policy would resolve to fmt=1 (int8-row), "
            f"not fmt=8, on an unstamped read (THE DESIGN LANDMINE, colibri.c)")


def shard_inventory(path, n_layers):
    """--dry-run: scan one shard's header only (get_slice, no tensor data pulled)
    and return the list of tensors that WOULD be repacked, without writing
    anything. Cheap: safe to run over an entire real checkpoint's headers."""
    from safetensors import safe_open
    inv = []
    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())
        for name in f.keys():
            sl = f.get_slice(name)
            dt = sl.get_dtype()
            if not is_repack_target(name, dt, keys, n_layers):
                continue
            O, I = sl.get_shape()
            sc_slice = f.get_slice(name + "_scale_inv")
            nblkO, nblkI = sc_slice.get_shape()
            _check_geometry(name, O, I, nblkO, nblkI)
            inv.append({"name": name, "kind": classify(name, n_layers),
                       "O": O, "I": I, "nblkO": nblkO, "nblkI": nblkI,
                       "weight_bytes": O * I, "scale_bytes": nblkO * nblkI * 4})
    return inv


def repack_shard(path, n_layers):
    """Real mode: byte-preserve every selected tensor's weight bytes (raw e4m3,
    no dequant/requant -- `.view(torch.uint8)` is a pure bit-reinterpret, verified
    byte-identical against torch.float8_e4m3fn's own storage) and rename/flatten
    the `_scale_inv` sidecar to the engine's `name.qs` convention (flat F32,
    nblkO*nblkI elements, row-major [blkO,blkI] -- matching
    scale[(o/128)*nblkI+i/128] on the read side). Returns (out_dict, inventory,
    fmt_map): fmt_map is {weight_tensor_name: FORMAT_NAME} for every selected
    tensor -- the caller JSON-encodes it into the output shard's __metadata__
    (see the module docstring's METADATA STAMP section). Keyed by the WEIGHT
    name only (not the ".qs" sidecar) -- that's the name qt_resolve_fmt/
    qt_verify_fmt_stamp look the stamp up by on the read side."""
    import torch
    from safetensors import safe_open
    out = {}
    inv = []
    fmt_map = {}
    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())
        for name in f.keys():
            sl = f.get_slice(name)
            dt = sl.get_dtype()
            if not is_repack_target(name, dt, keys, n_layers):
                continue
            w = f.get_tensor(name)                       # torch.float8_e4m3fn [O,I]
            sc = f.get_tensor(name + "_scale_inv")        # torch.float32 [nblkO,nblkI]
            O, I = w.shape
            nblkO, nblkI = sc.shape
            _check_geometry(name, O, I, nblkO, nblkI)
            out[name] = w.view(torch.uint8).contiguous()              # BYTE-PRESERVED
            out[name + ".qs"] = sc.to(torch.float32).reshape(-1).contiguous()
            fmt_map[name] = FORMAT_NAME
            inv.append({"name": name, "kind": classify(name, n_layers),
                       "O": O, "I": I, "nblkO": nblkO, "nblkI": nblkI,
                       "weight_bytes": O * I, "scale_bytes": nblkO * nblkI * 4})
    return out, inv, fmt_map


def _print_inventory_summary(all_inv, dry_run):
    by_kind = {}
    tot_w = tot_s = 0
    for it in all_inv:
        by_kind.setdefault(it["kind"], []).append(it)
        tot_w += it["weight_bytes"]; tot_s += it["scale_bytes"]
    tag = "[DRY-RUN]" if dry_run else "[REPACK]"
    print(f"{tag} {len(all_inv)} tensor(s) selected across all shards "
         f"({tot_w/1e9:.3f} GB weights, {tot_s/1e6:.3f} MB scales)")
    for kind in sorted(by_kind):
        items = by_kind[kind]
        w = sum(it["weight_bytes"] for it in items)
        print(f"{tag}   kind={kind:5s} {len(items):5d} tensor(s)  {w/1e9:.3f} GB")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", required=True, help="directory of Z.ai FP8 *.safetensors shards")
    ap.add_argument("--outdir", required=True, help="destination for repacked fmt=8 container shards")
    ap.add_argument("--n-layers", type=int, default=78)
    ap.add_argument("--dry-run", action="store_true",
                    help="inventory only: scan headers, print selection counts, write nothing")
    a = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(a.indir, "*.safetensors")))
    if not shards:
        print(f"ERROR: no *.safetensors files in {a.indir}"); sys.exit(1)

    if a.dry_run:
        all_inv = []
        for sp in shards:
            all_inv.extend(shard_inventory(sp, a.n_layers))
        _print_inventory_summary(all_inv, dry_run=True)
        return

    os.makedirs(a.outdir, exist_ok=True)
    # #383-class guard, reused (not reimplemented): refuses a resumed run whose
    # params differ from what's already partially in this outdir (the #355 failure
    # mode -- a second pass with different flags silently mixing containers).
    # Owns its own sidecar (.fp8pass-params.json), separate from the shard-progress
    # manifest below, exactly like the --repo mtp/indexer loops in
    # convert_fp8_to_int4.py use it.
    params = {"mode": "fp8-passthrough", "n_layers": a.n_layers}
    if not check_or_record_params(a.outdir, "fp8pass-", params):
        sys.exit(1)

    # RESUME (same #383-class manifest idiom as convert_fp8_to_int4.py's --indir path):
    # a sidecar records input-shard -> output-shard-name-or-"" (shards with no
    # resident-FP8 tensors emit no file), written atomically so an interrupted run
    # never leaves a half-written manifest for the next invocation to trust. The
    # params guard above already owns "don't mix conversions" -- this manifest is
    # purely which shards are done.
    prog_path = os.path.join(a.outdir, ".fp8pass-progress.json")
    prog = {}
    if os.path.exists(prog_path):
        try:
            prog = json.loads(open(prog_path).read())
        except (OSError, ValueError):
            prog = {}
    done = prog.setdefault("shards", {})

    from safetensors.torch import save_file
    all_inv = []
    n = fresh = skipped = 0
    for sp in shards:
        key = os.path.basename(sp)
        prev = done.get(key)
        if prev is not None and (prev == "" or os.path.exists(os.path.join(a.outdir, prev))):
            if prev:
                n += 1
            skipped += 1
            continue
        out, inv, fmt_map = repack_shard(sp, a.n_layers)
        all_inv.extend(inv)
        if not out:
            done[key] = ""
        else:
            name = f"out-fp8pass-{n:05d}.safetensors"
            # METADATA STAMP: __metadata__ values must be strings (safetensors
            # constraint) -- colibri.fmt's VALUE is itself JSON text (a map of
            # tensor name -> format NAME), not a nested object, for exactly
            # that reason. sort_keys for a deterministic, diffable stamp.
            meta = {METADATA_KEY: json.dumps(fmt_map, sort_keys=True)}
            save_file(out, os.path.join(a.outdir, name), metadata=meta)
            done[key] = name
            n += 1
            fresh += 1
        tmp = prog_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(prog, fh, indent=1)
        os.replace(tmp, prog_path)
    if skipped:
        print(f"[RESUME] {skipped} shard(s) already done in {a.outdir}, skipped")
    print(f"[REPACK] {fresh} new output shard(s) written")
    _print_inventory_summary(all_inv, dry_run=False)
    # FIX ROUND 2 (clean-room conformance trial, spec I6 -- loud failure, every
    # refusal names its condition): a run that selects ZERO repack-target tensors
    # across every shard under --indir (present run AND any prior resumed run --
    # `done` has one entry per shard, "" meaning that shard has NEVER produced an
    # output) used to exit 0 having emitted nothing -- a silent trap: an empty
    # "container" (just the resume/params sidecars, no actual .safetensors output)
    # that nobody asked for and no caller-side check would catch. Refuse loudly
    # instead. --dry-run is NOT covered here: reporting "0 tensors selected" IS
    # the loud, honest answer dry-run exists to give, not a silent no-op -- see
    # the early `return` above, before any of this resume/write bookkeeping.
    if all(v == "" for v in done.values()):
        print(f"ERROR: no repack-target tensors found under {a.indir} (checked "
              f"{len(shards)} shard(s), 0 selected) -- nothing emitted; refusing "
              f"to exit 0 for an empty container nobody asked for", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
