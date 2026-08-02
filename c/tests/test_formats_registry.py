"""docs/FORMATS.md <-> colibri.c FMT_NAMES parity (fix round 1, adopted
review test-candidate).

The registry's name column IS the canonical stamp string: a container's
__metadata__["colibri.fmt"] entry is matched byte-for-byte against
FMT_NAMES by qt_fmt_by_name(), and FORMATS.md is what a third-party tool
author will read to learn the string to write. A divergence between the
two (the class of defect this test pins: the registry shipped "e8-iq3"
while FMT_NAMES said "e8-iq3-lattice") means a tool stamping the
documented name produces containers the reader REFUSES as unrecognized.

Mechanism: regex over the two authoritative TEXTS themselves (the C
source and the markdown), not a generated header. Justification: a
generated header is a third copy that can go stale exactly like the doc
did, plus build machinery for one test; the C table's syntax is a
trivially stable one-entry-per-line initializer and the doc's table is
the reviewed artifact itself. Both parses assert plausibility floors so
an anchor drifting (renamed array, retitled section) fails loudly as a
parse error, never as a silent empty-vs-empty pass.

Parity rules:
  1. Every FMT_NAMES (name, fmt) entry must appear in the doc table as a
     row whose ordinal matches and whose FIRST backticked token in the
     name cell equals the C name exactly.
  2. Every doc table row whose ordinal has NO FMT_NAMES entry must say so
     explicitly ("no in-tree name string" -- the MXFP4 row's marker), so
     a new registry row can't silently claim a name the reader won't
     recognize.
"""
import os
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
C_DIR = os.path.normpath(os.path.join(HERE, ".."))
COLIBRI_C = os.path.join(C_DIR, "colibri.c")
FORMATS_MD = os.path.normpath(os.path.join(C_DIR, "..", "docs", "FORMATS.md"))


def parse_fmt_names(path):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    m = re.search(r"FMT_NAMES\[\]\s*=\s*\{(.*?)\};", src, re.S)
    if not m:
        raise AssertionError("FMT_NAMES[] initializer not found in colibri.c "
                             "(anchor drifted -- update this test's regex)")
    entries = re.findall(r'\{\s*"([^"]+)"\s*,\s*(\d+)\s*\}', m.group(1))
    return {int(fmt): name for name, fmt in entries}


def parse_doc_rows(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        m = re.match(r"\|\s*\*{0,2}(\d+)\*{0,2}\s*\|\s*(.+)$", line)
        if not m:
            continue
        ordinal = int(m.group(1))
        name_cell = m.group(2).split("|")[0]
        first_tick = re.search(r"`([^`]+)`", name_cell)
        rows[ordinal] = (first_tick.group(1) if first_tick else None, name_cell)
    return rows


class RegistryParityTest(unittest.TestCase):
    def setUp(self):
        self.c_names = parse_fmt_names(COLIBRI_C)
        self.doc_rows = parse_doc_rows(FORMATS_MD)
        # plausibility floors: an anchor drift must fail HERE, loudly.
        self.assertGreaterEqual(len(self.c_names), 8,
                                "FMT_NAMES parse found suspiciously few entries")
        self.assertGreaterEqual(len(self.doc_rows), 9,
                                "FORMATS.md table parse found suspiciously few rows")

    def test_every_c_name_matches_its_doc_row(self):
        for fmt, name in sorted(self.c_names.items()):
            self.assertIn(fmt, self.doc_rows,
                          f"FMT_NAMES has fmt={fmt} ('{name}') but FORMATS.md has no row for it")
            doc_name, cell = self.doc_rows[fmt]
            self.assertEqual(doc_name, name,
                             f"fmt={fmt}: FORMATS.md names '{doc_name}' but FMT_NAMES "
                             f"(the string qt_fmt_by_name actually matches) says '{name}'")

    def test_doc_rows_without_c_entry_are_marked(self):
        for fmt, (doc_name, cell) in sorted(self.doc_rows.items()):
            if fmt in self.c_names:
                continue
            self.assertIn("no in-tree name string", cell,
                          f"FORMATS.md row fmt={fmt} ('{doc_name}') has no FMT_NAMES entry "
                          f"and is not marked 'no in-tree name string' -- a tool stamping "
                          f"this documented name would be refused as unrecognized")


if __name__ == "__main__":
    unittest.main()
