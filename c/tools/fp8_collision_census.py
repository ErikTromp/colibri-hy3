#!/usr/bin/env python3
"""fp8_collision_census.py: enumerate every RESIDENT-role tensor shape in a
safetensors container (JSON header parse ONLY -- no tensor data read, no
torch/safetensors import, no model load) and evaluate THE DESIGN LANDMINE's
collision predicate (qt_resolve_fmt in colibri.c): O == ceil(O/128)*ceil(I/128),
the shape family where an int8-row tensor's per-row scale byte count (O*4)
exactly equals a per-128x128-block fp8 scale byte count
(ceil(O/128)*ceil(I/128)*4) for the SAME [O,I].

Maintainer review, #528: this is not a hypothetical corner case. GLM-5.2's own
self_attn.o_proj.weight ([D,H*v_head] = [6144,16384]) hits it on every
checkpoint this engine has loaded. This script is the "enumerated, not
sampled" evidence for that finding, run against this repo's own containers.

WHY config.json, not just the safetensors header: a container's on-disk
weight tensor is stored FLATTENED (one dimension: O*I raw bytes for int8-row,
O*ceil(I/2) for int4-packed, etc. -- see colibri.c's qt_bytes()) -- the header
alone cannot recover O and I separately for [O,I]. This script instead
computes each tensor's [O,I] from the model's config.json (hidden_size,
q_lora_rank, kv_lora_rank, num_attention_heads, qk_rope_head_dim,
qk_nope_head_dim, v_head_dim, intermediate_size, moe_intermediate_size -- the
same dimensions colibri.c's own Cfg struct reads at load time and passes into
qt_resolve_fmt as O,I), for every tensor NAME the header actually contains
that this script's role table recognizes, then cross-checks the computed
byte count against the header's own raw byte count (int8-row: O*I; int4-
packed: O*ceil(I/2)) as a sanity check -- reported, not silently trusted.

Scope: RESIDENT-kind tensor roles only (o_proj, q_a/q_b_proj,
kv_a_proj_with_mqa, kv_b_proj, dense-MLP gate/up/down, shared-expert
gate/up/down) plus routed-expert gate/up/down for completeness -- these are
the only roles tools/repack_fp8_passthrough.py's RESIDENT_KINDS classify()
taxonomy could ever select for fmt=8 repacking (routed experts are excluded
by that tool's own design and stay on int4-g64/E8, but are included here
anyway since the collision predicate is a pure function of [O,I], independent
of current on-disk format). f32/norm/router/embed/lm_head tensors are not
quantized (no U8 weight) and are skipped.

Usage:
  python3 tools/fp8_collision_census.py <container_dir> [<container_dir> ...]
"""
import glob, json, os, struct, sys

BLOCK = 128


def _nblk(n):
    return (n + BLOCK - 1) // BLOCK


def is_ambiguous(O, I):
    """THE DESIGN LANDMINE's collision predicate (colibri.c qt_resolve_fmt):
    an int8-row tensor at this [O,I] has per-row scale bytes O*4 that exactly
    equal per-128x128-block fp8 scale bytes ceil(O/128)*ceil(I/128)*4."""
    return _nblk(O) * _nblk(I) == O


def read_header(path):
    """Header-only read: 8-byte little-endian length prefix + that many bytes
    of JSON. Never reads a single byte of actual tensor data."""
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(hlen).decode("utf-8"))


def scan_tensor_headers(container_dir):
    """Union of every non-metadata tensor key -> header entry, across every
    shard in the container. Header (JSON) only, see read_header above."""
    entries = {}
    for path in sorted(glob.glob(os.path.join(container_dir, "*.safetensors"))):
        hdr = read_header(path)
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            entries[k] = v
    return entries


def classify_shape(name, cfg):
    """Returns (kind, O, I) for a tensor name this census can size from
    config.json alone, or None if `name` isn't a role this script recognizes
    (norms/router/embed/lm_head/MTP-head-only tensors -- none of these carry
    a resident [O,I] weight this predicate applies to)."""
    D = cfg["hidden_size"]
    H = cfg["num_attention_heads"]
    q_lora = cfg["q_lora_rank"]
    kv_lora = cfg["kv_lora_rank"]
    qk_rope = cfg["qk_rope_head_dim"]
    qk_nope = cfg["qk_nope_head_dim"]
    v_head = cfg["v_head_dim"]
    inter = cfg["intermediate_size"]
    moe_inter = cfg["moe_intermediate_size"]
    qk_head = qk_nope + qk_rope

    if name.endswith("self_attn.o_proj.weight"):
        return ("self_attn.o_proj", D, H * v_head)
    if name.endswith("self_attn.q_a_proj.weight"):
        return ("self_attn.q_a_proj", q_lora, D)
    if name.endswith("self_attn.q_b_proj.weight"):
        return ("self_attn.q_b_proj", H * qk_head, q_lora)
    if name.endswith("self_attn.kv_a_proj_with_mqa.weight"):
        return ("self_attn.kv_a_proj_with_mqa", kv_lora + qk_rope, D)
    if name.endswith("self_attn.kv_b_proj.weight"):
        return ("self_attn.kv_b_proj (kvb -- excluded from fmt=8 repack, see tool docstring)",
                 H * (qk_nope + v_head), kv_lora)
    if ".mlp.experts." in name and name.endswith("gate_proj.weight"):
        return ("mlp.experts.*.gate_proj (routed -- excluded from fmt=8 repack)", moe_inter, D)
    if ".mlp.experts." in name and name.endswith("up_proj.weight"):
        return ("mlp.experts.*.up_proj (routed -- excluded from fmt=8 repack)", moe_inter, D)
    if ".mlp.experts." in name and name.endswith("down_proj.weight"):
        return ("mlp.experts.*.down_proj (routed -- excluded from fmt=8 repack)", D, moe_inter)
    if "shared_experts" in name and name.endswith("gate_proj.weight"):
        return ("mlp.shared_experts.gate_proj", moe_inter, D)
    if "shared_experts" in name and name.endswith("up_proj.weight"):
        return ("mlp.shared_experts.up_proj", moe_inter, D)
    if "shared_experts" in name and name.endswith("down_proj.weight"):
        return ("mlp.shared_experts.down_proj", D, moe_inter)
    if name.endswith("mlp.gate_proj.weight") and ".experts." not in name:
        return ("mlp.gate_proj (dense)", inter, D)
    if name.endswith("mlp.up_proj.weight") and ".experts." not in name:
        return ("mlp.up_proj (dense)", inter, D)
    if name.endswith("mlp.down_proj.weight") and ".experts." not in name:
        return ("mlp.down_proj (dense)", D, inter)
    return None


def _byte_check(nb_actual, O, I):
    if nb_actual == O * I:
        return "int8-row (nb=O*I)"
    if nb_actual == O * ((I + 1) // 2):
        return "int4-packed (nb=O*ceil(I/2))"
    return f"UNVERIFIED (nb={nb_actual}, neither O*I={O*I} nor O*ceil(I/2)={O*((I+1)//2)} -- " \
           f"likely int3-g64/E8-lattice, expected for routed experts in some containers)"


def census(container_dir):
    cfg_path = os.path.join(container_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    entries = scan_tensor_headers(container_dir)
    by_pattern = {}
    unresolved = []
    for name, entry in entries.items():
        if name.endswith(".qs") or name.endswith("_scale_inv"):
            continue
        if entry.get("dtype") != "U8":
            continue
        cls = classify_shape(name, cfg)
        if cls is None:
            unresolved.append(name)
            continue
        kind, O, I = cls
        shp = entry.get("shape", [])
        nb_actual = 1
        for d in shp:
            nb_actual *= d
        key = (kind, O, I)
        rec = by_pattern.setdefault(key, {"kind": kind, "O": O, "I": I, "count": 0,
                                          "ambiguous": is_ambiguous(O, I),
                                          "byte_encodings": set()})
        rec["count"] += 1
        rec["byte_encodings"].add(_byte_check(nb_actual, O, I))
    return cfg, by_pattern, unresolved


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print("usage: fp8_collision_census.py <container_dir> [<container_dir> ...]", file=sys.stderr)
        sys.exit(1)
    any_ambig = False
    for d in dirs:
        base = os.path.basename(os.path.normpath(d))
        cfg, by_pattern, unresolved = census(d)
        print(f"=== {base} ({d}) ===")
        print(f"{'tensor name pattern':70s} {'O':>8s} {'I':>8s} {'nblkO*nblkI':>12s} {'count':>6s}  ambiguous?  on-disk encoding(s)")
        for (kind, O, I), rec in sorted(by_pattern.items()):
            nblk_prod = _nblk(O) * _nblk(I)
            flag = "AMBIGUOUS <---" if rec["ambiguous"] else "-"
            if rec["ambiguous"]:
                any_ambig = True
            enc = ", ".join(sorted(rec["byte_encodings"]))
            print(f"{kind:70s} {O:8d} {I:8d} {nblk_prod:12d} {rec['count']:6d}  {flag:15s} {enc}")
        if unresolved:
            print(f"  ({len(unresolved)} U8 tensor(s) in {base} not classified by this script's role "
                  f"table -- inspect if unexpected: {sorted(unresolved)[:5]}{'...' if len(unresolved)>5 else ''})")
        print()
    print("RESULT:", "AMBIGUOUS family found (see table above)" if any_ambig else "no ambiguous family found")


if __name__ == "__main__":
    main()
