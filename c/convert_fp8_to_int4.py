"""
Convertitore GLM-5.2-FP8 -> nostro container int4 (STADIO B).

Strategia DISK-SAFE (richiesta dell'utente): scarica UNO shard (~5 GB), lo converte in
int4, lo CANCELLA, passa al prossimo. Il disco non si riempie mai: picco = 1 shard + l'output
int4 che cresce fino a ~372 GB. Controllo di spazio che si ferma se manca margine.

Cosa fa per ogni tensore:
  - pesi FP8 (e4m3) con `*.weight_scale_inv`  -> dequant a blocchi 128x128 -> f32
  - pesi BF16 (norme/embed/lm_head/...)        -> f32
  poi:
  - attn/mlp/shared/expert/embed/lm_head -> QUANTIZZATO int4 (o int8) con la STESSA matematica
    del motore C (np.rint = lrintf, stesse soglie, stesso packing dei nibble) -> token identici
  - norme / router (mlp.gate.weight) / bias / e_score_correction_bias -> tenuti F32
  - indexer DSA / layer MTP (78) / shared_head / eh_proj / *norm dell'indexer -> SALTATI

Output: una dir di safetensors leggibile dal motore C (per ogni peso quantizzato: `nome` U8 =
dati impacchettati, `nome.qs` F32 = scale per riga).

USO:
  # test locale (oracolo tiny, niente download): converte una dir gia' presente
  python3 convert_fp8_to_int4.py --indir glm_tiny --outdir glm_tiny_i4 --ebits 4 --io-bits 4
  # selftest del dequant fp8 (richiede torch)
  python3 convert_fp8_to_int4.py --selftest
  # reale: scarica+converte+cancella shard per shard
  python3 convert_fp8_to_int4.py --repo zai-org/GLM-5.2-FP8 --outdir /home/vincenzo/glm52_i4
"""
import os, sys, glob, json, shutil, argparse
import numpy as np

# ---------- quantizzazione: identica al C (glm.c) ----------
def quant_int8(w, bits):                       # w: [O,I] f32 -> (qbytes U8 [O*I], scale f32 [O])
    qmax = (1 << (bits - 1)) - 1
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(w / s), -qmax - 1, qmax).astype(np.int8)
    return q.reshape(-1).view(np.uint8).copy(), s[:, 0].astype(np.float32)

def quant_int4(w, bits):                        # -> (qbytes U8 [O*ceil(I/2)], scale f32 [O])
    O, I = w.shape
    qmax = (1 << (bits - 1)) - 1
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(w / s), -8, qmax).astype(np.int32)  # nibble [-8,7]
    rb = (I + 1) // 2
    out = np.zeros((O, rb), np.uint8)
    v0 = (q[:, 0::2] + 8).astype(np.uint8)
    out[:, :v0.shape[1]] = v0
    if I > 1:
        v1 = (q[:, 1::2] + 8).astype(np.uint8)
        out[:, :v1.shape[1]] |= (v1 << 4)
    return out.reshape(-1), s[:, 0].astype(np.float32)

def quant_int2(w, bits):                        # -> (qbytes U8 [O*ceil(I/4)], scale f32 [O]); 4/byte
    O, I = w.shape
    qmax = (1 << (bits - 1)) - 1                 # bits=2 -> qmax=1, valori [-2,1]
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(w / s), -2, qmax).astype(np.int32)
    rb = (I + 3) // 4
    out = np.zeros((O, rb), np.uint8)
    for k in range(4):                           # impacchetta 4 valori per byte (identico a pack_int2 in C)
        vk = q[:, k::4]
        out[:, :vk.shape[1]] |= ((vk + 2).astype(np.uint8) << (k * 2))
    return out.reshape(-1), s[:, 0].astype(np.float32)

# ---------- classificazione dei tensori ----------
def layer_idx(name):
    p = name.split(".")
    if len(p) > 2 and p[0] == "model" and p[1] == "layers":
        try: return int(p[2])
        except ValueError: return -1
    return -1

def classify(name, n_layers, keep_mtp=False, keep_idx=False):
    if name.endswith("_scale_inv"): return "consumed"   # gestito col suo peso
    li = layer_idx(name)
    if keep_idx:
        # modalita' --indexer: SOLO i pesi del DSA lightning indexer dei layer principali
        if li < 0 or li >= n_layers or "indexer" not in name: return "skip"
        if name.endswith("norm.weight"): return "f32"
        return "q"                                       # int8 consigliato (--ebits 8): pesi di scoring
    if keep_mtp:
        if li != n_layers: return "skip"                 # solo il layer MTP
        if "indexer" in name: return "skip"              # il DSA indexer resta un no-op
    else:
        if li >= n_layers: return "skip"                 # layer MTP (78)
        if any(k in name for k in ["indexer", "indexers_proj", "eh_proj",
                                    "enorm", "hnorm", "shared_head"]): return "skip"
    if name.endswith("e_score_correction_bias"): return "f32"
    if name.endswith("mlp.gate.weight"): return "f32"    # router (NON gate_proj)
    if name.endswith("norm.weight") or name == "model.norm.weight": return "f32"
    if name in ("model.embed_tokens.weight", "lm_head.weight"): return "io"
    if ".mlp.experts." in name and name.endswith(".weight"): return "x"  # expert ROUTED (streaming)
    if name.endswith(".weight"): return "q"              # attn/dense-mlp/shared (residente)
    return "f32"

# ---------- dequant di un tensore (fp8+scale a blocchi / bf16 / f32) ----------
def dequant(f, name):
    import torch
    sl = f.get_slice(name); dt = sl.get_dtype()
    if dt in ("F8_E4M3", "float8_e4m3fn"):
        w = f.get_tensor(name).to(torch.float32)
        sc = f.get_tensor(name + "_scale_inv").to(torch.float32)   # [ceil(O/128),ceil(I/128)]
        O, I = w.shape
        sc = sc.repeat_interleave(128, 0).repeat_interleave(128, 1)[:O, :I]
        return (w * sc).numpy()
    return f.get_tensor(name).to(torch.float32).numpy()

def convert_shard(path, out_dict, n_layers, ebits, io_bits, xbits, keep_mtp=False, keep_idx=False):
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        for name in f.keys():
            kind = classify(name, n_layers, keep_mtp, keep_idx)
            if kind in ("skip", "consumed"): continue
            w = dequant(f, name)
            if kind == "f32":
                out_dict[name] = w.astype(np.float32)
            else:
                bits = io_bits if kind == "io" else xbits if kind == "x" else ebits
                if w.ndim != 2:        # es. bias 1D non previsto come 'q' -> tienilo f32
                    out_dict[name] = w.astype(np.float32); continue
                q, s = (quant_int2(w, bits) if bits <= 2 else
                        quant_int4(w, bits) if bits <= 4 else quant_int8(w, bits))
                out_dict[name] = q
                out_dict[name + ".qs"] = s

def free_gb(p): return shutil.disk_usage(p).free / 1e9

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--indir", default=None)
    ap.add_argument("--outdir", required=False)
    ap.add_argument("--ebits", type=int, default=4)      # bit residenti: attn/dense-mlp/shared
    ap.add_argument("--io-bits", type=int, default=8)    # bit di embed/lm_head
    ap.add_argument("--xbits", type=int, default=None)   # bit degli expert ROUTED (streaming); default=ebits
    ap.add_argument("--n-layers", type=int, default=78)
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mtp", action="store_true",
        help="scarica/converte SOLO la testa MTP (model.layers.<n_layers>.*) -> out-mtp-*.safetensors")
    ap.add_argument("--indexer", action="store_true",
        help="estrae SOLO i pesi del DSA lightning indexer -> out-idx-*.safetensors. ATTENZIONE: "
             "i tensori indexer sono sparsi su ~tutti gli shard: ri-scarica l'intero repo (~756 GB "
             "di traffico) per tenerne pochi GB. Resumabile shard per shard. Consigliato --ebits 8.")
    a = ap.parse_args()
    if a.xbits is None: a.xbits = a.ebits

    if a.selftest:
        import torch
        w = (torch.randn(256, 256) * 0.3)
        O, I = w.shape; bs = 128
        sc = torch.zeros(O // bs, I // bs)
        for bi in range(O // bs):
            for bj in range(I // bs):
                blk = w[bi*bs:(bi+1)*bs, bj*bs:(bj+1)*bs]
                sc[bi, bj] = blk.abs().max() / 448.0
        q = (w / sc.repeat_interleave(bs,0).repeat_interleave(bs,1)).to(torch.float8_e4m3fn)
        deq = (q.to(torch.float32) * sc.repeat_interleave(bs,0).repeat_interleave(bs,1))
        rel = (deq - w).abs().mean() / w.abs().mean()
        print(f"[selftest fp8 block-dequant] errore relativo medio = {rel:.4f}  "
              f"({'OK' if rel < 0.05 else 'ALTO'})")
        return

    os.makedirs(a.outdir, exist_ok=True)
    if a.indir:    # conversione locale (test)
        shards = sorted(glob.glob(os.path.join(a.indir, "*.safetensors")))
        from safetensors.numpy import save_file
        for i, sp in enumerate(shards):
            out = {}; convert_shard(sp, out, a.n_layers, a.ebits, a.io_bits, a.xbits)
            save_file(out, os.path.join(a.outdir, f"out-{i:05d}.safetensors"))
        # copia config + tokenizer
        for fn in ["config.json"]:
            src = os.path.join(a.indir, fn)
            if os.path.exists(src): shutil.copy(src, a.outdir)
        print(f"convertito {len(shards)} shard -> {a.outdir}")
        return

    # reale: scarica shard per shard, converte, cancella
    # ROBUSTEZZA RETE (WSL: la scheda virtuale puo' bloccarsi): timeout sulle read cosi' un
    # download appeso FALLISCE invece di restare fermo per sempre, e retry con backoff.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    # hf_xet si blocca quando la rete WSL viene riavviata (connessioni zombie senza timeout):
    # forza la via HTTP classica, che curl ha dimostrato funzionare. (misurato 2026-07-02)
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import HfApi, hf_hub_download

    # lock anti-doppione: DUE convertitori sulla stessa outdir si corrompono a vicenda
    import fcntl
    lock = open(os.path.join(a.outdir, ".convert.lock"), "w")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("ERRORE: un altro convertitore sta gia' lavorando su questa outdir. Esco."); return

    def download_retry(repo, fn, dest, tries=999):
        import time as _t
        for att in range(tries):
            try:
                return hf_hub_download(repo, fn, local_dir=dest)
            except KeyboardInterrupt: raise
            except Exception as ex:
                wait = min(60, 5 * (att + 1))
                print(f"    rete KO ({type(ex).__name__}): riprovo tra {wait}s "
                      f"(tentativo {att+1})", flush=True)
                _t.sleep(wait)
        raise RuntimeError("download fallito dopo troppi tentativi")

    from safetensors.numpy import save_file
    import time as _t
    for att in range(999):
        try:
            info = HfApi().repo_info(a.repo, files_metadata=True); break
        except KeyboardInterrupt: raise
        except Exception as ex:
            w = min(60, 5*(att+1)); print(f"repo_info KO ({type(ex).__name__}): riprovo tra {w}s", flush=True); _t.sleep(w)
    shards = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".safetensors"))
    for fn in ["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"]:
        try: shutil.copy(hf_hub_download(a.repo, fn, local_dir=a.outdir+"/_meta"), a.outdir)
        except Exception: pass
    tmp = os.path.join(a.outdir, "_inflight"); os.makedirs(tmp, exist_ok=True)
    if a.mtp:
        import urllib.request
        idx = json.loads(urllib.request.urlopen(
            f"https://huggingface.co/{a.repo}/resolve/main/model.safetensors.index.json", timeout=30).read())["weight_map"]
        pref = f"model.layers.{a.n_layers}."
        mtp_shards = sorted(set(v for k, v in idx.items() if k.startswith(pref)))
        print(f"[MTP] testa nel layer {a.n_layers}: {len(mtp_shards)} shard da processare: {mtp_shards}")
        for i, sh in enumerate(mtp_shards):
            outp = os.path.join(a.outdir, f"out-mtp-{i:05d}.safetensors")
            if os.path.exists(outp): print(f"[MTP] {outp} gia' fatto"); continue
            print(f"[MTP {i+1}/{len(mtp_shards)}] scarico {sh}...", flush=True)
            p = download_retry(a.repo, sh, tmp)
            out = {}; convert_shard(p, out, a.n_layers, a.ebits, a.io_bits, a.xbits, keep_mtp=True)
            save_file(out, outp)
            os.remove(p)
            for blob in glob.glob(os.path.join(tmp, "**", "*"), recursive=True):
                if os.path.isfile(blob): os.remove(blob)
            print(f"    -> {os.path.basename(outp)} ({os.path.getsize(outp)/1e9:.2f} GB, {len(out)} tensori)", flush=True)
        shutil.rmtree(tmp, ignore_errors=True); print("[MTP] FATTO."); return
    if a.indexer:
        import urllib.request
        idx = json.loads(urllib.request.urlopen(
            f"https://huggingface.co/{a.repo}/resolve/main/model.safetensors.index.json", timeout=30).read())["weight_map"]
        idx_shards = sorted(set(v for k, v in idx.items()
                                if "indexer" in k and 0 <= layer_idx(k) < a.n_layers))
        tot_gb = len(idx_shards) * 5.4
        print(f"[IDX] pesi indexer su {len(idx_shards)} shard (~{tot_gb:.0f} GB di download totale, resumabile)")
        for i, sh in enumerate(idx_shards):
            outp = os.path.join(a.outdir, f"out-idx-{i:05d}.safetensors")
            if os.path.exists(outp): continue             # gia' fatto -> ripartibile
            print(f"[IDX {i+1}/{len(idx_shards)}] scarico {sh}...", flush=True)
            p = download_retry(a.repo, sh, tmp)
            out = {}; convert_shard(p, out, a.n_layers, a.ebits, a.io_bits, a.xbits, keep_idx=True)
            if out: save_file(out, outp)
            os.remove(p)
            for blob in glob.glob(os.path.join(tmp, "**", "*"), recursive=True):
                if os.path.isfile(blob): os.remove(blob)
            print(f"    -> {os.path.basename(outp)} ({len(out)} tensori)", flush=True)
        shutil.rmtree(tmp, ignore_errors=True); print("[IDX] FATTO."); return
    for i, sh in enumerate(shards):
        if free_gb(a.outdir) < a.min_free_gb:
            print(f"STOP: spazio libero < {a.min_free_gb} GB. Libera spazio e rilancia (riprende)."); break
        outp = os.path.join(a.outdir, f"out-{i:05d}.safetensors")
        if os.path.exists(outp): continue                 # gia' fatto -> ripartibile
        print(f"[{i+1}/{len(shards)}] scarico {sh} (libero {free_gb(a.outdir):.0f} GB)...", flush=True)
        p = download_retry(a.repo, sh, tmp)
        out = {}; convert_shard(p, out, a.n_layers, a.ebits, a.io_bits, a.xbits)
        save_file(out, outp)
        os.remove(p)                                       # <-- cancella subito lo shard fp8
        for blob in glob.glob(os.path.join(tmp, "**", "*"), recursive=True):
            if os.path.isfile(blob): os.remove(blob)
        print(f"    -> {os.path.basename(outp)} ({os.path.getsize(outp)/1e9:.2f} GB)", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print("FATTO." if i == len(shards)-1 else "INTERROTTO (rilancia per riprendere).")

if __name__ == "__main__":
    main()
