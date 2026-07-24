#!/usr/bin/env python3
"""Generate the fmt=6 kernel fixture from the reference codec (#452).

tests/test_e8_kernel.c checks the C kernel against these bytes, so the fixture
must come from iq3_pack.py — the format's reference implementation — and never
from the kernel itself. Regenerate whenever the layout changes:

    python3 tools/make_e8_fixture.py
"""
import os
import struct

import numpy as np

import iq3_pack as P

O, I = 24, 512
SEED = 20260721


def main():
    rng = np.random.default_rng(SEED)
    w = (rng.standard_normal((O, I)) * 0.05).astype(np.float32)
    packed = P.encode(w)
    deq = P.decode(packed, I).astype(np.float32)
    x = (rng.standard_normal(I) * 1.0).astype(np.float32)
    # float64 reference so the C float path is compared against something better
    yref = (deq.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)

    # Rotated section (#452 converter step): the container stores W@Q; the
    # engine rotates activations with e8_rot_rows. I2=768 tiles as 256+512,
    # exercising the multi-block path and the regenerated signs on both sides.
    O2, I2 = 16, 768
    w2 = (rng.standard_normal((O2, I2)) * 0.05).astype(np.float32)
    packed2 = P.encode(P.rotate_rows(w2))
    x2 = (rng.standard_normal(I2) * 1.0).astype(np.float32)
    xrot2 = P.rotate_rows(x2[None, :])[0].astype(np.float32)
    y2ref = (P.decode(packed2, I2).astype(np.float64)
             @ xrot2.astype(np.float64)).astype(np.float32)

    out = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "e8_case.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", O, I))
        f.write(packed.tobytes())
        f.write(deq.tobytes())
        f.write(x.tobytes())
        f.write(yref.tobytes())
        f.write(struct.pack("<ii", O2, I2))
        f.write(packed2.tobytes())
        f.write(x2.tobytes())
        f.write(xrot2.tobytes())
        f.write(y2ref.tobytes())
    print(f"wrote {path}: O={O} I={I} packed={packed.nbytes}B "
          f"({packed.nbytes * 8 / (O * I):.4f} bpw) + rotated O2={O2} I2={I2}")


if __name__ == "__main__":
    main()
