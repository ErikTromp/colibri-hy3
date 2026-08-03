"""tools/repack_fp8_passthrough.py: fmt=8 repack tool tests.

fmt=8 is a PUBLIC ordinal: see repack_fp8_passthrough.py's module docstring
for the fmt=6 -> fmt=100 -> fmt=7 -> fmt=8 history (dev's own #465 merged a
REAL fmt=6, E8/IQ3, so this tool's format moved to the PRIVATE ORDINAL BLOCK
as fmt=100 during development; the maintainer assigned fmt=7 on #524; #705
then merged MXFP4 as fmt=7 while this PR was open, forcing the renumber
to fmt=8).

Synthetic fixtures ONLY (tools/glm_fp8_emit.py, the exact real-checkpoint FP8
layout that convert_fp8_to_int4.py's dequant() reads) -- no real Z.ai shard is
read or written by this suite, per the build's hard constraint. Covers:
selection (resident kinds byte-preserved, routed experts / io / f32 excluded),
byte-for-byte preservation of the fp8 weight and the .qs scale rename, the
scale-geometry refusal path in TWO forms (THE DESIGN LANDMINE's write-side
twin) -- a malformed source shard (.qs shape doesn't match ceil(O/128)x
ceil(I/128) for its weight) must be refused, AND a well-formed-but-AMBIGUOUS
shape (nblkO*nblkI==O, where this tensor's block-scale byte count would
coincide with a per-row int8 scale byte count) must ALSO be refused
(maintainer review, #528: this is not hypothetical, GLM-5.2's own
self_attn.o_proj.weight is a real instance) -- --dry-run writing nothing, and
the #383-class resume/params-guard behavior. FIX ROUND 2 (clean-room
conformance trial, spec I6): a real (non-dry-run) run that selects ZERO
repack-target tensors across --indir must refuse loudly (nonzero exit,
stderr naming the condition), not exit 0 having emitted nothing -- an empty
"container" nobody asked for; --dry-run's own "0 selected" report is
unaffected (that IS the loud, honest answer dry-run exists to give).
"""
import glob, json, os, struct, subprocess, sys, tempfile, unittest

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError as e:
    raise unittest.SkipTest(f"torch/safetensors not installed: {e}")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from glm_fp8_emit import save_fp8_safetensors
import repack_fp8_passthrough as rp


def _read_header(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(hlen))


def _make_fixture(path, D=256, I_=384, E=4):
    """One synthetic shard covering every selection case: SELECTED resident
    kinds (attn/o/sh/dmlp), the EXCLUDED resident kind kvb (kv_b_proj -- valid
    per convert_fp8_to_int4.classify(), but the engine's CPU/CUDA MLA-absorb
    paths have no fmt=8 case yet, so this tool must not select it -- see the
    module docstring), routed experts (must be excluded), the router and a
    norm (f32-kept, never fp8), and embed_tokens (io kind -- excluded by kind
    even though glm_fp8_emit's simpler keep_f32() happens to FP8-quantize it,
    since it doesn't know the full classify() taxonomy; that mismatch is
    itself a useful case: is_repack_target must exclude by KIND, not by
    guessing at source dtype conventions)."""
    torch.manual_seed(0)
    sd = {
        "model.layers.0.self_attn.o_proj.weight": torch.randn(D, D) * 0.02,
        "model.layers.0.self_attn.q_a_proj.weight": torch.randn(D, D) * 0.02,
        "model.layers.0.self_attn.kv_b_proj.weight": torch.randn(D, D) * 0.02,
        "model.layers.0.mlp.shared_experts.gate_proj.weight": torch.randn(D, I_) * 0.02,
        "model.layers.0.mlp.gate_proj.weight": torch.randn(D, I_) * 0.02,       # dmlp
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(I_, D) * 0.02,  # routed: EXCLUDE
        "model.layers.0.mlp.experts.1.up_proj.weight": torch.randn(I_, D) * 0.02,    # routed: EXCLUDE
        "model.layers.0.input_layernorm.weight": torch.randn(D),                # f32: EXCLUDE
        "model.layers.0.mlp.gate.weight": torch.randn(E, D),                    # router f32: EXCLUDE
        "model.embed_tokens.weight": torch.randn(64, D),                        # io: EXCLUDE
    }
    n_fp8, n_tot = save_fp8_safetensors(sd, path)
    return sd, n_fp8, n_tot


class SelectionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.indir = os.path.join(self.tmp.name, "fp8src")
        os.makedirs(self.indir)
        self.shard = os.path.join(self.indir, "model-00001-of-00001.safetensors")
        self.sd, self.n_fp8, self.n_tot = _make_fixture(self.shard)

    def tearDown(self):
        self.tmp.cleanup()

    def test_dry_run_selection_and_no_writes(self):
        inv = rp.shard_inventory(self.shard, n_layers=5)
        names = {it["name"] for it in inv}
        self.assertEqual(names, {
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.q_a_proj.weight",
            "model.layers.0.mlp.shared_experts.gate_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
        })
        kinds = {it["name"]: it["kind"] for it in inv}
        self.assertEqual(kinds["model.layers.0.self_attn.o_proj.weight"], "o")
        self.assertEqual(kinds["model.layers.0.self_attn.q_a_proj.weight"], "attn")
        self.assertEqual(kinds["model.layers.0.mlp.shared_experts.gate_proj.weight"], "sh")
        self.assertEqual(kinds["model.layers.0.mlp.gate_proj.weight"], "dmlp")
        # --dry-run through the CLI must not create the outdir at all
        outdir = os.path.join(self.tmp.name, "out_dry")
        rc = os.system(
            f'{sys.executable} "{os.path.join(os.path.dirname(__file__), "..", "tools", "repack_fp8_passthrough.py")}" '
            f'--indir "{self.indir}" --outdir "{outdir}" --n-layers 5 --dry-run')
        self.assertEqual(rc, 0)
        self.assertFalse(os.path.exists(outdir), "--dry-run must not write anything")

    def test_routed_experts_excluded(self):
        inv = rp.shard_inventory(self.shard, n_layers=5)
        names = {it["name"] for it in inv}
        self.assertNotIn("model.layers.0.mlp.experts.0.gate_proj.weight", names)
        self.assertNotIn("model.layers.0.mlp.experts.1.up_proj.weight", names)

    def test_kv_b_proj_excluded(self):
        """Regression guard for a gap found in self-review: kv_b_proj is a valid
        resident kind per classify(), but colibri.c's CPU absorb path
        (qt_addrow/qt_matvec_rows) and the CUDA absorb kernels have no fmt=8
        case -- selecting it here would produce a container the engine
        silently misreads as int2. Must stay excluded until that path gains
        fmt=8 support (separate follow-up work)."""
        inv = rp.shard_inventory(self.shard, n_layers=5)
        names = {it["name"] for it in inv}
        self.assertNotIn("model.layers.0.self_attn.kv_b_proj.weight", names)
        self.assertNotIn("kvb", rp.RESIDENT_KINDS)

    def test_io_and_f32_excluded(self):
        inv = rp.shard_inventory(self.shard, n_layers=5)
        names = {it["name"] for it in inv}
        self.assertNotIn("model.embed_tokens.weight", names)          # io kind
        self.assertNotIn("model.layers.0.input_layernorm.weight", names)   # f32 (also not FP8 dtype)
        self.assertNotIn("model.layers.0.mlp.gate.weight", names)     # router, f32 kind

    def test_byte_preservation_and_qs_rename(self):
        out, inv, fmt_map = rp.repack_shard(self.shard, n_layers=5)
        self.assertEqual(len(inv), 4)
        with safe_open(self.shard, framework="pt") as f:
            for it in inv:
                name = it["name"]
                w_src = f.get_tensor(name).view(torch.uint8)
                sc_src = f.get_tensor(name + "_scale_inv").reshape(-1)
                self.assertTrue(torch.equal(out[name], w_src), f"{name}: weight bytes not preserved")
                self.assertEqual(out[name].dtype, torch.uint8)
                self.assertTrue(torch.equal(out[name + ".qs"], sc_src), f"{name}: scale not preserved")
                O, I_ = w_src.shape
                nblkO, nblkI = (O + 127) // 128, (I_ + 127) // 128
                self.assertEqual(out[name + ".qs"].numel(), nblkO * nblkI)

    def test_fmt_map_covers_exactly_selected_weight_names(self):
        """fmt_map (the METADATA STAMP payload) must cover exactly the selected
        WEIGHT names -- never the ".qs" sidecars, never an excluded tensor -- and
        every value must be FORMAT_NAME (the format's public NAME, not the
        internal fmt=8 ordinal)."""
        out, inv, fmt_map = rp.repack_shard(self.shard, n_layers=5)
        selected_names = {it["name"] for it in inv}
        self.assertEqual(set(fmt_map.keys()), selected_names)
        for name in selected_names:
            self.assertNotIn(name + ".qs", fmt_map)
            self.assertEqual(fmt_map[name], rp.FORMAT_NAME)
        self.assertEqual(rp.FORMAT_NAME, "fp8-e4m3-b128")

    def test_geometry_refusal_on_malformed_scale(self):
        """A shard whose _scale_inv shape doesn't match ceil(O/128)xceil(I/128) for
        its weight must be REFUSED (write-side twin of qt_resolve_fmt's read-side
        refusal for the same landmine) rather than silently repacked."""
        bad_path = os.path.join(self.tmp.name, "bad.safetensors")
        D = 300
        w = torch.randn(D, D).to(torch.float8_e4m3fn)
        bad_scale = torch.ones(1, 1, dtype=torch.float32)   # wrong: should be [3,3] for D=300
        # o_proj.weight is a resident "o"-kind name -> is_repack_target will select it
        save_file({"model.layers.0.self_attn.o_proj.weight": w,
                  "model.layers.0.self_attn.o_proj.weight_scale_inv": bad_scale}, bad_path)
        with self.assertRaises(ValueError):
            rp.repack_shard(bad_path, n_layers=5)
        with self.assertRaises(ValueError):
            rp.shard_inventory(bad_path, n_layers=5)

    def test_geometry_refusal_on_ambiguous_shape(self):
        """WRITER-SIDE REFUSAL (maintainer review, #528): a WELL-FORMED .qs sidecar
        (correct nblkOxnblkI shape for [O,I], unlike the malformed case above) must
        still be refused when nblkO*nblkI==O -- the exact shape where this tensor's
        block-scale byte count (nblkO*nblkI*4) would coincide with a per-row int8
        scale byte count (O*4). Not hypothetical: GLM-5.2's own
        self_attn.o_proj.weight ([6144,16384], nblkO=48 nblkI=128 product=6144==O)
        is a real instance of this exact family; O=2,I=256 here is the smallest
        realistic analog (nblkO=1, nblkI=2, product=2==O), the same degenerate
        shape family c/tests/test_fp8_load.c's test_disambiguation() sweeps. The
        engine's qt_resolve_fmt reader resolves an unstamped collision at this
        shape to fmt=1 (int8-row) rather than refusing (see colibri.c's
        INVERSION) -- so this tool refusing to ever EMIT an fmt=8 container here
        is what keeps that reader-side resolution correct: an fmt=8 tensor this
        tool produced at this shape would be silently read back as plain int8."""
        amb_path = os.path.join(self.tmp.name, "ambiguous.safetensors")
        O, I_ = 2, 256
        w = torch.randn(O, I_).to(torch.float8_e4m3fn)
        nblkO, nblkI = (O + 127) // 128, (I_ + 127) // 128
        self.assertEqual(nblkO * nblkI, O, "fixture must actually hit the collision")
        scale = torch.ones(nblkO, nblkI, dtype=torch.float32)   # WELL-FORMED for [O,I] -- passes the first check
        save_file({"model.layers.0.self_attn.o_proj.weight": w,
                  "model.layers.0.self_attn.o_proj.weight_scale_inv": scale}, amb_path)
        with self.assertRaises(ValueError):
            rp.repack_shard(amb_path, n_layers=5)
        with self.assertRaises(ValueError):
            rp.shard_inventory(amb_path, n_layers=5)


class ResumeAndParamsGuardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.indir = os.path.join(self.tmp.name, "fp8src")
        os.makedirs(self.indir)
        self.shard = os.path.join(self.indir, "model-00001-of-00001.safetensors")
        _make_fixture(self.shard)
        self.outdir = os.path.join(self.tmp.name, "out")
        self.tool = os.path.join(os.path.dirname(__file__), "..", "tools", "repack_fp8_passthrough.py")

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, n_layers):
        return os.system(f'{sys.executable} "{self.tool}" --indir "{self.indir}" '
                         f'--outdir "{self.outdir}" --n-layers {n_layers} >/dev/null 2>&1')

    def test_output_container_geometry(self):
        self.assertEqual(self._run(5), 0)
        outs = glob.glob(os.path.join(self.outdir, "out-fp8pass-*.safetensors"))
        self.assertEqual(len(outs), 1)
        hdr = _read_header(outs[0])
        # o_proj is [256,256] -> nblkO=nblkI=2 -> 4 block scales
        qs = hdr["model.layers.0.self_attn.o_proj.weight.qs"]
        self.assertEqual(qs["dtype"], "F32")
        self.assertEqual(qs["shape"], [4])
        w = hdr["model.layers.0.self_attn.o_proj.weight"]
        self.assertEqual(w["dtype"], "U8")
        self.assertEqual(w["shape"], [256, 256])
        # experts / io / f32 / kv_b_proj must be absent from the output container entirely
        self.assertNotIn("model.layers.0.mlp.experts.0.gate_proj.weight", hdr)
        self.assertNotIn("model.embed_tokens.weight", hdr)
        self.assertNotIn("model.layers.0.input_layernorm.weight", hdr)
        self.assertNotIn("model.layers.0.self_attn.kv_b_proj.weight", hdr)

    def test_metadata_stamp_present_and_correct(self):
        """End-to-end through the real CLI: __metadata__["colibri.fmt"] must be
        present, parse as JSON, and name exactly the selected resident-kind
        tensors -- the real round trip a unit-level rp.repack_shard() call
        can't exercise (save_file's own metadata= handling)."""
        self.assertEqual(self._run(5), 0)
        outs = glob.glob(os.path.join(self.outdir, "out-fp8pass-*.safetensors"))
        self.assertEqual(len(outs), 1)
        hdr = _read_header(outs[0])
        self.assertIn("__metadata__", hdr)
        self.assertIn("colibri.fmt", hdr["__metadata__"])
        stamp = json.loads(hdr["__metadata__"]["colibri.fmt"])
        self.assertEqual(set(stamp.keys()), {
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.q_a_proj.weight",
            "model.layers.0.mlp.shared_experts.gate_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
        })
        for name, fmt_name in stamp.items():
            self.assertEqual(fmt_name, rp.FORMAT_NAME, f"{name}: stamped format name mismatch")

    def test_resume_skips_completed_shard(self):
        self.assertEqual(self._run(5), 0)
        mtime1 = os.path.getmtime(glob.glob(os.path.join(self.outdir, "out-fp8pass-*.safetensors"))[0])
        self.assertEqual(self._run(5), 0)                 # rerun, same params
        outs = glob.glob(os.path.join(self.outdir, "out-fp8pass-*.safetensors"))
        self.assertEqual(len(outs), 1, "resume must not duplicate output shards")
        self.assertEqual(os.path.getmtime(outs[0]), mtime1, "resume must not rewrite a completed shard")

    def test_params_mismatch_refused(self):
        self.assertEqual(self._run(5), 0)
        rc = self._run(78)                                 # different n_layers, same outdir
        self.assertNotEqual(rc, 0, "a params mismatch on the same outdir must be refused")
        # and the FIRST run's output must be untouched by the refused second run
        outs = glob.glob(os.path.join(self.outdir, "out-fp8pass-*.safetensors"))
        self.assertEqual(len(outs), 1)


def _make_zero_target_fixture(path, D=256, I_=384):
    """A shard with real FP8 tensors, but NONE matching a repack-target kind --
    mirrors the trial's own scenario (a non-resident-named FP8 tensor selects
    nothing): one routed expert (kind "x", explicitly excluded) and one f32
    norm (never a repack candidate at all)."""
    torch.manual_seed(2)
    sd = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(I_, D) * 0.02,  # routed: EXCLUDE
        "model.layers.0.input_layernorm.weight": torch.randn(D),                     # f32: EXCLUDE
    }
    save_fp8_safetensors(sd, path)


class ZeroTargetRefusalTest(unittest.TestCase):
    """FIX ROUND 2, item 3 (clean-room conformance trial, spec I6): a real
    (non-dry-run) run whose --indir has FP8 tensors but none matching a
    repack-target kind must refuse loudly, not exit 0 having written nothing."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.indir = os.path.join(self.tmp.name, "fp8src")
        os.makedirs(self.indir)
        self.shard = os.path.join(self.indir, "model-00001-of-00001.safetensors")
        _make_zero_target_fixture(self.shard)
        self.outdir = os.path.join(self.tmp.name, "out")
        self.tool = os.path.join(os.path.dirname(__file__), "..", "tools", "repack_fp8_passthrough.py")

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, extra_args=()):
        return subprocess.run(
            [sys.executable, self.tool, "--indir", self.indir, "--outdir", self.outdir,
             "--n-layers", "5", *extra_args],
            capture_output=True, text=True)

    def test_zero_targets_refuses_loudly(self):
        proc = self._run()
        self.assertNotEqual(proc.returncode, 0, "zero repack-target tensors must refuse, not exit 0")
        self.assertIn("no repack-target tensors", proc.stderr)
        self.assertIn(self.indir, proc.stderr, "the refusal must name the --indir condition")
        # confirmed empty: no output container was (or should be) produced
        self.assertEqual(glob.glob(os.path.join(self.outdir, "out-fp8pass-*.safetensors")), [])

    def test_zero_targets_dry_run_unaffected(self):
        """--dry-run's own "0 tensor(s) selected" report is the loud, honest
        answer dry-run exists to give -- not a silent no-op -- so it must NOT
        be turned into a refusal by this fix."""
        proc = self._run(["--dry-run"])
        self.assertEqual(proc.returncode, 0)
        self.assertIn("0 tensor(s) selected", proc.stdout)


if __name__ == "__main__":
    unittest.main()
