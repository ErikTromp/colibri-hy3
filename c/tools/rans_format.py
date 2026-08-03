"""int4-rans256-g0 format reference module, shared by repack_rans.py (writer)
and rans_verify.py (validator).

This is the Python twin of c/rans.h — the same codec, the same chunk-record
framing, the same named refusal classes. When the optional C bridge library
(`make rans` -> tools/librans_c.*) is present it does the heavy per-stream
work; when absent, the pure-Python fallback here produces byte-identical
output far more slowly (the e2e test pins the two implementations equal, the
same relationship iq3_pack.py has with tools/libiq3).

FORMAT SUMMARY (docs/int4-rans256-g0.md is the full specification):
  - a weight tensor's bytes are ONE chunk record: n_symbols u64,
    packed_bytes u64, stream_offsets[N+1] u32, zero-pad to 16, payload
    (N=256 independent rANS streams, round-robin symbol assignment j -> j%N),
    zero-pad to 16;
  - the shared static table ships once per shard in
    __metadata__["colibri.int4-rans256-g0.table"] (JSON: table_id, n_streams,
    scale_bits, M, freq[16], start[16], slot_to_symbol_b64, table_crc32);
  - __metadata__["colibri.fmt"] maps each entropy-coded WEIGHT tensor name to
    the format NAME "int4-rans256-g0". THE STAMP IS MANDATORY: entropy-coded
    sizes are data-dependent, so no byte-arithmetic inference exists for this
    format — the stamp is the ONLY signal a U8 tensor is entropy-coded, not a
    cross-check on an inference (unlike fmt=8's optional stamp).

Table construction is the standard largest-remainder frequency quantization
(FSE/rANS convention): scale an empirical histogram to sum exactly to
M = 2**scale_bits, keep every present symbol encodable (freq >= 1), hand out
the remainder by largest fractional part.
"""
import base64
import ctypes
import hashlib
import json
import os
import struct
import zlib

import numpy as np

FORMAT_NAME = "int4-rans256-g0"
N_STREAMS = 256
SCALE_BITS = 14                # M = 16384; generous for a 16-symbol alphabet
M = 1 << SCALE_BITS
TABLE_ID = "g0"                # shared-global-table generation 0
METADATA_KEY = "colibri.fmt"   # same key/shape as the fmt=8 stamp convention
TABLE_KEY = "colibri.int4-rans256-g0.table"
RANS_L = 1 << 23
SLACK = 64                     # readable bytes past a record buffer (SIMD contract)


class RansRefusal(Exception):
    """A named TRUST-VERIFY-REFUSE failure. `code` matches c/rans.h's
    rans_err_name strings (E_TRUNCATED, E_TABLE_CRC, ...) plus the
    container-level classes only the Python tools can see (stamps, metadata)."""

    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


# ---------------------------------------------------------------------------
# table construction (ported unchanged from the proven prototype)
# ---------------------------------------------------------------------------
def quantize_freq(hist, m=M):
    """hist: length-16 array of raw counts. Returns freq[16] uint32 summing to
    m; largest-remainder quantization; min 1 count for any nonzero-count
    symbol (so it stays encodable); absent symbols get exactly 0.

    Determinism: both argsorts use kind="stable" so exact fractional-remainder
    (or frequency) ties break by symbol index identically on every host and
    numpy version — contract 4's cross-host byte-identity depends on it.

    Termination: requires m >= the number of present symbols (raises
    ValueError otherwise — the min-1 constraint would be unsatisfiable).
    Given that, the deficit<0 loop provably terminates: while deficit < 0,
    sum(freq) = m - deficit > m >= n_present, so some entry exceeds 1 and
    every full pass over `order` decrements at least one entry."""
    hist = np.asarray(hist, dtype=np.float64)
    total = hist.sum()
    assert total > 0
    n_present = int(np.count_nonzero(hist))
    if m < n_present:
        raise ValueError(
            f"quantize_freq: m={m} < {n_present} present symbols — every "
            f"present symbol needs freq >= 1, so this table size is impossible")
    raw = hist / total * m
    freq = np.floor(raw).astype(np.int64)
    nonzero = hist > 0
    freq[nonzero & (freq == 0)] = 1
    deficit = m - freq.sum()
    if deficit > 0:
        remainder = raw - np.floor(raw)
        remainder[~nonzero] = -1     # never give slots to symbols absent from data
        order = np.argsort(-remainder, kind="stable")
        i = 0
        while deficit > 0 and i < len(order):
            idx = order[i]
            if nonzero[idx]:
                freq[idx] += 1
                deficit -= 1
            i += 1
    elif deficit < 0:
        order = np.argsort(-freq, kind="stable")
        i = 0
        while deficit < 0:
            idx = order[i % len(order)]
            if freq[idx] > 1:
                freq[idx] -= 1
                deficit += 1
            i += 1
    assert freq.sum() == m, (freq.sum(), m)
    return freq.astype(np.uint32)


def build_table(freq, m=M):
    start = np.zeros(len(freq), dtype=np.uint32)
    start[1:] = np.cumsum(freq)[:-1].astype(np.uint32)
    slot_to_symbol = np.zeros(m, dtype=np.uint16)
    for s, (f, st) in enumerate(zip(freq, start)):
        if f:
            slot_to_symbol[st:st + f] = s
    return start, slot_to_symbol


def table_blob(freq, start, slot_to_symbol):
    """The per-shard __metadata__ table value (as a dict; the writer
    json.dumps(..., sort_keys=True) it — deterministic, diffable)."""
    slot_bytes = np.ascontiguousarray(slot_to_symbol, dtype="<u2").tobytes()
    return {
        "table_id": TABLE_ID,
        "n_streams": N_STREAMS,
        "scale_bits": SCALE_BITS,
        "M": M,
        "freq": [int(x) for x in freq],
        "start": [int(x) for x in start],
        "slot_to_symbol_b64": base64.b64encode(slot_bytes).decode("ascii"),
        "table_crc32": f"{zlib.crc32(slot_bytes):08x}",
    }


def parse_table_blob(raw):
    """Validate + decode the metadata table value. Raises RansRefusal with a
    named class for every malformation the break-it battery covers."""
    try:
        blob = json.loads(raw)
    except ValueError as exc:
        raise RansRefusal("E_TABLE_MALFORMED", f"table metadata is not JSON: {exc}")
    for key in ("table_id", "n_streams", "scale_bits", "M", "freq", "start",
                "slot_to_symbol_b64", "table_crc32"):
        if key not in blob:
            raise RansRefusal("E_TABLE_MALFORMED", f"table metadata lacks '{key}'")
    sb = blob["scale_bits"]
    if not isinstance(sb, int) or not (1 <= sb <= 15):
        raise RansRefusal("E_TABLE_SCALE", f"scale_bits={sb!r} out of range 1..15")
    m = blob["M"]
    if m != (1 << sb):
        raise RansRefusal("E_TABLE_SCALE", f"M={m!r} != 1<<scale_bits={1 << sb}")
    if blob["n_streams"] != N_STREAMS:
        raise RansRefusal("E_TABLE_MALFORMED",
                          f"n_streams={blob['n_streams']!r} (this format fixes 256)")
    freq = blob["freq"]
    start = blob["start"]
    if len(freq) != 16 or len(start) != 16 or \
            not all(isinstance(x, int) and 0 <= x <= m for x in freq + start):
        raise RansRefusal("E_TABLE_MALFORMED", "freq/start must be 16 ints each")
    if sum(freq) != m:
        raise RansRefusal("E_TABLE_FREQ_SUM", f"freq sums to {sum(freq)}, not M={m}")
    acc = 0
    for s in range(16):
        if start[s] != acc:
            raise RansRefusal("E_TABLE_START", f"start[{s}]={start[s]} != prefix {acc}")
        acc += freq[s]
    try:
        slot_bytes = base64.b64decode(blob["slot_to_symbol_b64"], validate=True)
    except Exception as exc:
        raise RansRefusal("E_TABLE_MALFORMED", f"slot_to_symbol_b64: {exc}")
    if len(slot_bytes) != 2 * m:
        raise RansRefusal("E_TABLE_MALFORMED",
                          f"slot_to_symbol is {len(slot_bytes)} bytes, want {2 * m}")
    crc = f"{zlib.crc32(slot_bytes):08x}"
    if crc != blob["table_crc32"]:
        raise RansRefusal("E_TABLE_CRC",
                          f"slot table crc {crc} != stamped {blob['table_crc32']}")
    slot = np.frombuffer(slot_bytes, dtype="<u2")
    freq_a = np.asarray(freq, dtype=np.uint32)
    start_a = np.asarray(start, dtype=np.uint32)
    counts = np.bincount(slot, minlength=17)
    if slot.max(initial=0) > 15 or counts[:16].tolist() != freq or \
            not np.array_equal(np.sort(slot), slot):
        # sortedness + per-symbol counts == freq pins the slot table to be
        # exactly the canonical start/freq banding
        raise RansRefusal("E_TABLE_SLOT", "slot_to_symbol inconsistent with freq[]")
    return {"scale_bits": sb, "M": m, "freq": freq_a, "start": start_a,
            "slot_to_symbol": np.ascontiguousarray(slot, dtype=np.uint16),
            "table_id": blob["table_id"]}


# ---------------------------------------------------------------------------
# nibble packing (low-nibble-first, the container convention)
# ---------------------------------------------------------------------------
def unpack_nibbles(raw_bytes):
    b = np.frombuffer(raw_bytes, dtype=np.uint8)
    out = np.empty(b.size * 2, dtype=np.uint8)
    out[0::2] = b & 0x0F
    out[1::2] = b >> 4
    return out


def pack_nibbles(nibbles):
    n = nibbles.reshape(-1, 2)
    return ((n[:, 0] | (n[:, 1] << 4)).astype(np.uint8)).tobytes()


# ---------------------------------------------------------------------------
# optional C bridge (make rans)
# ---------------------------------------------------------------------------
def _load_lib():
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("librans_c.dylib", "librans_c.so", "rans_c.dll"):
        path = os.path.join(here, name)
        if not os.path.exists(path):
            continue
        lib = ctypes.CDLL(path)
        u8p = ctypes.POINTER(ctypes.c_uint8)
        u16p = ctypes.POINTER(ctypes.c_uint16)
        u32p = ctypes.POINTER(ctypes.c_uint32)
        u64p = ctypes.POINTER(ctypes.c_uint64)
        lib.rc_record_encode.argtypes = [u8p, ctypes.c_uint64, u32p, u32p, u16p,
                                         ctypes.c_uint32, ctypes.c_uint32,
                                         u8p, ctypes.c_uint64]
        lib.rc_record_encode.restype = ctypes.c_int64
        lib.rc_record_bound.argtypes = [ctypes.c_uint64, ctypes.c_uint32]
        lib.rc_record_bound.restype = ctypes.c_uint64
        lib.rc_record_parse.argtypes = [u8p, ctypes.c_uint64, ctypes.c_uint32,
                                        u64p, u64p]
        lib.rc_record_parse.restype = ctypes.c_int32
        lib.rc_record_decode.argtypes = [u8p, ctypes.c_uint64, ctypes.c_uint32,
                                         u32p, u32p, u16p, ctypes.c_uint32,
                                         ctypes.c_int32, u8p]
        lib.rc_record_decode.restype = ctypes.c_int32
        lib.rc_stream_decode_checked.argtypes = [u8p, ctypes.c_uint64,
                                                 ctypes.c_uint64, u32p, u32p,
                                                 u16p, ctypes.c_uint32, u8p]
        lib.rc_stream_decode_checked.restype = ctypes.c_int32
        lib.rc_err_name.argtypes = [ctypes.c_int32]
        lib.rc_err_name.restype = ctypes.c_char_p
        return lib
    return None


LIB = None if os.environ.get("RANS_FORCE_PYTHON") else _load_lib()


def _p8(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


def _p16(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))


def _p32(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))


def _err_name(code):
    return LIB.rc_err_name(code).decode("ascii")


# ---------------------------------------------------------------------------
# pure-Python codec fallback (bit-exact twin of c/rans.h's scalar paths)
# ---------------------------------------------------------------------------
def _py_encode_stream(nibbles, freq, start, scale_bits):
    out = bytearray()                      # collected backwards, reversed at the end
    x = RANS_L
    for s in nibbles[::-1]:
        s = int(s)
        f = int(freq[s])
        x_max = ((RANS_L >> scale_bits) << 8) * f
        while x >= x_max:
            out.append(x & 0xFF)
            x >>= 8
        x = ((x // f) << scale_bits) + (x % f) + int(start[s])
    for _ in range(4):
        out.append(x & 0xFF)
        x >>= 8
    return bytes(out[::-1])


def _py_decode_stream_checked(buf, n, freq, start, slot_to_symbol, scale_bits):
    """Returns (nibbles ndarray, refusal-name-or-None); mirrors
    rans_decode_stream_checked in c/rans.h refusal for refusal."""
    if len(buf) < 4:
        return None, "E_STREAM_UNDERRUN"
    mask = (1 << scale_bits) - 1
    pos = 4
    x = int.from_bytes(buf[:4], "big")
    if x < RANS_L:
        return None, "E_STREAM_STATE_RANGE"
    end = len(buf)
    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        slot = x & mask
        s = int(slot_to_symbol[slot])
        x = int(freq[s]) * (x >> scale_bits) + slot - int(start[s])
        while x < RANS_L:
            if pos >= end:
                break
            x = (x << 8) | buf[pos]
            pos += 1
        out[i] = s
    if pos != end:
        return None, "E_STREAM_LEFTOVER"
    if x != RANS_L:
        return None, "E_STREAM_FINAL_STATE"
    return out, None


# ---------------------------------------------------------------------------
# chunk record: build / parse / decode
# ---------------------------------------------------------------------------
def _round16(v):
    return (v + 15) & ~15


def record_header_bytes(n_streams=N_STREAMS):
    return _round16(16 + (n_streams + 1) * 4)


def build_record(nibbles, freq, start, slot_to_symbol, scale_bits=SCALE_BITS,
                 n_streams=N_STREAMS):
    """nibbles -> one chunk-record bytes object. Deterministic."""
    nib = np.ascontiguousarray(nibbles, dtype=np.uint8)
    n = nib.size
    if n_streams != N_STREAMS:
        raise RansRefusal("E_NSTREAMS",
                          f"n_streams={n_streams} (this format fixes {N_STREAMS})")
    if n == 0:
        raise RansRefusal("E_EMPTY", "refusing to encode an empty tensor")
    # pre-scan parity with c/rans.h: refuse uncodable input by name before
    # any encoding work (values > 15, or symbols the table gives freq 0)
    freq_a = np.asarray(freq, dtype=np.uint32)
    if int(nib.max(initial=0)) > 15 or (freq_a[nib] == 0).any():
        raise RansRefusal("E_SYMBOL_UNCODABLE",
                          "input contains a value > 15 or a symbol whose "
                          "table frequency is zero")
    if LIB is not None:
        cap = int(LIB.rc_record_bound(n, n_streams)) + SLACK
        out = np.zeros(cap, dtype=np.uint8)
        freq_a = np.ascontiguousarray(freq, dtype=np.uint32)
        start_a = np.ascontiguousarray(start, dtype=np.uint32)
        slot_a = np.ascontiguousarray(slot_to_symbol, dtype=np.uint16)
        rc = LIB.rc_record_encode(_p8(nib), n, _p32(freq_a), _p32(start_a),
                                  _p16(slot_a), scale_bits, n_streams,
                                  _p8(out), cap)
        if rc < 0:
            raise RansRefusal(_err_name(int(rc)), "rc_record_encode failed")
        return out[:rc].tobytes()
    # pure-Python fallback: identical construction
    streams = [_py_encode_stream(nib[i::n_streams], freq, start, scale_bits)
               for i in range(n_streams)]
    offsets = np.zeros(n_streams + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum([len(s) for s in streams]).astype(np.uint32)
    payload = b"".join(streams)
    head = struct.pack("<QQ", n, (n + 1) // 2) + offsets.astype("<u4").tobytes()
    head += b"\x00" * (_round16(len(head)) - len(head))
    payload += b"\x00" * (_round16(len(payload)) - len(payload))
    return head + payload


def parse_record(blob, n_streams=N_STREAMS):
    """Validate framing; returns dict(n_symbols, packed_bytes, offsets,
    payload). Raises RansRefusal with the same named classes as c/rans.h —
    that parity is a TESTED property (the e2e suite feeds identical bytes to
    both implementations and asserts identical classes)."""
    if n_streams == 0:
        raise RansRefusal("E_NSTREAMS", "n_streams == 0")
    head = record_header_bytes(n_streams)
    if len(blob) < head:
        raise RansRefusal("E_TRUNCATED", f"{len(blob)} bytes < {head}-byte header")
    n_symbols, packed_bytes = struct.unpack_from("<QQ", blob, 0)
    if n_symbols == 0:
        raise RansRefusal("E_EMPTY", "n_symbols == 0")
    # Python ints do not wrap, so ceil(n/2) here already agrees with the C
    # side's non-wrapping n/2 + (n&1) for every uint64 value incl. UINT64_MAX
    if packed_bytes != (n_symbols + 1) // 2:
        raise RansRefusal("E_COUNT_MISMATCH",
                          f"packed_bytes={packed_bytes} != ceil({n_symbols}/2)")
    offsets = np.frombuffer(blob, dtype="<u4", count=n_streams + 1, offset=16)
    if offsets[0] != 0:
        raise RansRefusal("E_OFFSET_FIRST", f"stream_offsets[0]={offsets[0]}")
    d = np.diff(offsets.astype(np.int64))
    if (d < 0).any():
        raise RansRefusal("E_OFFSETS_MONOTONIC", "stream_offsets not non-decreasing")
    if (d < 4).any():
        raise RansRefusal("E_STREAM_SHORT",
                          f"stream {int(np.argmax(d < 4))} shorter than its 4-byte state")
    payload_len = int(offsets[n_streams])
    if head + payload_len > len(blob):
        raise RansRefusal("E_TRUNCATED",
                          f"payload claims {payload_len} bytes past the blob end")
    if len(blob) != head + _round16(payload_len):
        raise RansRefusal("E_LENGTH_MISMATCH",
                          f"blob is {len(blob)} bytes, framing derives "
                          f"{head + _round16(payload_len)}")
    # amplification bound, identical to c/rans.h: no table this format admits
    # can encode more than payload_len*8*M_max symbols into this payload —
    # refuse the decompression bomb before anything sizes buffers from
    # n_symbols
    if n_symbols > payload_len * 8 * (1 << 15):
        raise RansRefusal("E_OVERSIZE",
                          f"n_symbols={n_symbols} impossibly large for a "
                          f"{payload_len}-byte payload")
    raw_head_end = 16 + (n_streams + 1) * 4
    if any(blob[raw_head_end:head]) or any(blob[head + payload_len:]):
        raise RansRefusal("E_PAD_NONZERO", "derived padding bytes are not zero")
    return {"n_symbols": n_symbols, "packed_bytes": packed_bytes,
            "offsets": offsets, "payload": blob[head:head + payload_len]}


def decode_record(blob, table, n_streams=N_STREAMS, checked=True):
    """Record bytes -> original packed bytes. `table` is parse_table_blob's
    result (or an equivalent dict). Raises RansRefusal on any malformation,
    including the stream-level invariants a genuine encoder output always
    satisfies (state range / exact consumption / final state)."""
    rec = parse_record(blob, n_streams)
    freq = np.ascontiguousarray(table["freq"], dtype=np.uint32)
    start = np.ascontiguousarray(table["start"], dtype=np.uint32)
    slot = np.ascontiguousarray(table["slot_to_symbol"], dtype=np.uint16)
    sb = table["scale_bits"]
    n = rec["n_symbols"]
    # parse_record bounded n_symbols against the payload (E_OVERSIZE); a
    # record that legitimately passes can still be too big for THIS machine —
    # that is a named refusal, never an allocator traceback
    try:
        nib = np.empty(n, dtype=np.uint8)
        out = np.empty(rec["packed_bytes"], dtype=np.uint8)
    except MemoryError:
        raise RansRefusal("E_NOMEM",
                          f"cannot allocate buffers for n_symbols={n}")
    if LIB is not None and not checked:
        buf = np.zeros(len(blob) + SLACK, dtype=np.uint8)   # SLACK contract
        buf[:len(blob)] = np.frombuffer(blob, dtype=np.uint8)
        rc = LIB.rc_record_decode(_p8(buf), len(blob), n_streams, _p32(freq),
                                  _p32(start), _p16(slot), sb, -1, _p8(out))
        if rc != 0:
            raise RansRefusal(_err_name(rc), "rc_record_decode failed")
        return out.tobytes()
    # checked per-stream decode (verification path), C-accelerated per stream
    payload = rec["payload"]
    offsets = rec["offsets"]
    for i in range(n_streams):
        sub = payload[int(offsets[i]):int(offsets[i + 1])]
        n_i = len(range(i, n, n_streams))
        if LIB is not None:
            sbuf = np.frombuffer(sub, dtype=np.uint8)
            sout = np.empty(max(n_i, 1), dtype=np.uint8)
            rc = LIB.rc_stream_decode_checked(_p8(np.ascontiguousarray(sbuf)),
                                              len(sub), n_i, _p32(freq),
                                              _p32(start), _p16(slot), sb,
                                              _p8(sout))
            if rc != 0:
                raise RansRefusal(_err_name(rc), f"stream {i} failed checked decode")
            dec = sout[:n_i]
        else:
            dec, err = _py_decode_stream_checked(sub, n_i, freq, start, slot, sb)
            if err:
                raise RansRefusal(err, f"stream {i} failed checked decode")
        nib[i::n_streams] = dec
    if n % 2:
        nib = np.concatenate([nib, np.zeros(1, dtype=np.uint8)])
    return pack_nibbles(nib)


# ---------------------------------------------------------------------------
# minimal safetensors i/o (header-only reads; deterministic writes) — no
# external safetensors dependency, same spirit as doctor.py's own parser
# ---------------------------------------------------------------------------
def read_header(path):
    """Parse AND validate a shard header. An untrusted container reaches the
    tools through this function, so the whole header shape is pinned here —
    every malformation is a named E_SHARD_MALFORMED refusal, never a
    downstream KeyError/TypeError traceback: the header must be a JSON
    object; every tensor entry an object with a string dtype, a list shape,
    and data_offsets = [b, e] with 0 <= b <= e <= data size; __metadata__
    (when present) a string-to-string object."""
    fsize = os.path.getsize(path)
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise RansRefusal("E_SHARD_MALFORMED", f"{path}: no header length")
        hlen = struct.unpack("<Q", raw)[0]
        if hlen > 512 * 1024 * 1024 or 8 + hlen > fsize:
            raise RansRefusal("E_SHARD_MALFORMED", f"{path}: absurd header ({hlen} B)")
        try:
            header = json.loads(f.read(hlen))
        except ValueError as exc:
            raise RansRefusal("E_SHARD_MALFORMED", f"{path}: header not JSON: {exc}")
    if not isinstance(header, dict):
        raise RansRefusal("E_SHARD_MALFORMED",
                          f"{path}: header is {type(header).__name__}, not an object")
    data_start = 8 + hlen
    data_size = fsize - data_start
    for name, info in header.items():
        if name == "__metadata__":
            if not isinstance(info, dict) or \
                    not all(isinstance(k, str) and isinstance(v, str)
                            for k, v in info.items()):
                raise RansRefusal("E_SHARD_MALFORMED",
                                  f"{path}: __metadata__ is not a str->str object")
            continue
        if not isinstance(info, dict):
            raise RansRefusal("E_SHARD_MALFORMED",
                              f"{path}:{name}: tensor entry is not an object")
        if not isinstance(info.get("dtype"), str) or \
                not isinstance(info.get("shape"), list):
            raise RansRefusal("E_SHARD_MALFORMED",
                              f"{path}:{name}: missing/malformed dtype or shape")
        off = info.get("data_offsets")
        if (not isinstance(off, list) or len(off) != 2 or
                not all(isinstance(x, int) and not isinstance(x, bool) for x in off) or
                not (0 <= off[0] <= off[1] <= data_size)):
            raise RansRefusal("E_SHARD_MALFORMED",
                              f"{path}:{name}: data_offsets {off!r} invalid for "
                              f"{data_size} data bytes")
    return header, data_start


def read_tensor_bytes(path, header, data_start, name):
    info = header[name]
    b, e = info["data_offsets"]          # validated by read_header
    with open(path, "rb") as f:
        f.seek(data_start + b)
        raw = f.read(e - b)
    if len(raw) != e - b:
        raise RansRefusal("E_SHARD_MALFORMED", f"{path}:{name}: short read")
    return raw, info


def write_shard(path, tensors, metadata):
    """tensors: list of (name, dtype, shape, bytes) — written in the given
    order; metadata: {str: str}. Fully deterministic byte layout (sorted
    metadata keys, insertion-ordered tensors, no timestamps). Returns the
    sha256 hex of the COMPLETE file bytes, computed while streaming the
    write (before the atomic rename) — the whole-file digest the manifest
    records."""
    header = {"__metadata__": {k: metadata[k] for k in sorted(metadata)}}
    offset = 0
    for name, dtype, shape, raw in tensors:
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    hjson = json.dumps(header, separators=(",", ":"), sort_keys=False).encode()
    h = hashlib.sha256()
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        for chunk in (struct.pack("<Q", len(hjson)), hjson):
            f.write(chunk)
            h.update(chunk)
        for _, _, _, raw in tensors:
            f.write(raw)
            h.update(raw)
    os.replace(tmp, path)
    return h.hexdigest()


def sha256_file(path, bufsize=1 << 20):
    """sha256 hex of a whole file, chunked (shards can be GB-scale)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(bufsize)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
