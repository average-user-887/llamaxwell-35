#!/usr/bin/env python3
"""NC2 CPU-only tests (no GPU, no docker). usage: NC2_DIR=<dir with harness+analyzer> python3 -B test_nc2.py
Default NC2_DIR = this file's dir (fixed code). Point NC2_DIR at the G1c evidence dir to see the bounds test FAIL on the original."""
import os, sys, re, tempfile, unittest, importlib.util, subprocess
sys.dont_write_bytecode = True
D = os.environ.get('NC2_DIR', os.path.dirname(os.path.abspath(__file__)))
NV = 50; os.environ['G1C_NV'] = str(NV); os.environ['G1C_NR'] = os.environ.get('NC2_NR', '128')
def imp(name):
    sp = importlib.util.spec_from_file_location(name, os.path.join(D, name + '.py')); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
H = imp('g1c_harness')
import numpy as np
def max_pos_needed(h):
    """highest token index the C++ tool will read, from the harness's own event list"""
    ev = h.events([0], h.BUCKETS, None); pos = 0; mx = 0
    for e in ev:
        if e[0] == 'P': a, z = map(int, e[1:].split(':')); assert a == pos, 'P start %d != pos %d' % (a, pos); pos = z; mx = max(mx, z - 1)
        elif e[0] == 'R': n = int(e[1:]); mx = max(mx, pos + n - 1); pos += n
    return mx
class Bounds(unittest.TestCase):
    def tokens_available(self):
        return H.doc_ntok() if hasattr(H, 'doc_ntok') else H.DOCLEN   # original harness slices exactly DOCLEN tokens
    def test_128_rows_inside_doc_slice(self):
        self.assertEqual(H.NR, 128)
        need = max_pos_needed(H) + 1
        self.assertLessEqual(need, self.tokens_available(), 'last row reads token %d but doc slice has only %d tokens (out of bounds)' % (need - 1, self.tokens_available()))
    def test_64_rows_unchanged(self):
        # 64 rows must fit the original 98368 slice (arm A/B vectors stay valid)
        self.assertEqual(98304 + 64, H.DOCLEN)
    @unittest.skipUnless(hasattr(H, 'validate_plan'), 'original has no validate_plan')
    def test_validate_plan_rejects_short_fixture(self):
        with self.assertRaises(SystemExit): H.validate_plan([7], (32768, 65536, 98304), 128, 7 * H.DOCLEN + H.DOCLEN)
        H.validate_plan([7], (32768, 65536, 98304), 128, 7 * H.DOCLEN + H.DOCLEN + 64)
A = imp('g1c_analyze')
NR = A.NR
def mk(out, arm, docs=(0, 1), buckets=(32768, 65536, 98304), seed=0, eps=0.0):
    os.makedirs('%s/%s' % (out, arm), exist_ok=True)
    for k in docs:
        for B in buckets:
            base = np.random.default_rng(k * 7 + B).normal(0, 3, (NR, NV)).astype(np.float32)
            (base + eps * np.random.default_rng(seed + k).normal(0, 1, base.shape).astype(np.float32)).tofile('%s/%s/doc%d_b%d.f32' % (out, arm, k, B))
@unittest.skipUnless(hasattr(A, 'AnalyzerError'), 'original analyzer has no validation')
class Hardening(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory(); self.o = self.t.name; os.environ['G1C_DOCS'] = '0,1'
        mk(self.o, 'A'); mk(self.o, 'B', seed=1, eps=0.001)
    def tearDown(self): self.t.cleanup()
    def err(self, pat):
        with self.assertRaisesRegex(A.AnalyzerError, pat): A.main(self.o)
    def test_ok(self): A.main(self.o); self.assertTrue(os.path.exists(self.o + '/G1C_TABLE.md'))
    def test_missing_doc(self): os.remove(self.o + '/B/doc1_b65536.f32'); self.err('partial run.*B/doc1_b65536')
    def test_empty_file(self): open(self.o + '/A/doc0_b32768.f32', 'wb').close(); self.err('partial run.*A/doc0_b32768')
    def test_missing_arm(self):
        import shutil; shutil.rmtree(self.o + '/B'); self.err('partial run')
    def test_fewer_rows(self):
        f = self.o + '/A/doc0_b98304.f32'; d = open(f, 'rb').read()[:(NR - 1) * NV * 4]
        with open(f, 'wb') as g: g.write(d)
        self.err('size.*expected.*partial or wrong row count')
    def test_wrong_vocab(self):
        np.zeros((NR, NV + 3), np.float32).tofile(self.o + '/B/doc0_b32768.f32'); self.err('not a multiple of vocab|expected')
    def test_wrong_vocab_same_total(self):
        # same total size but env vocab differs -> rows*vocab mismatch caught by size check
        old = A.NV; A.NV = NV * 2
        try: self.err('size')
        finally: A.NV = old
    def test_nan(self):
        a = np.fromfile(self.o + '/B/doc0_b32768.f32', np.float32); a[5] = np.nan; a.tofile(self.o + '/B/doc0_b32768.f32'); self.err('B/doc0_b32768.*non-finite.*row 0 col 5')
    def test_inf(self):
        a = np.fromfile(self.o + '/A/doc1_b65536.f32', np.float32); a[NV + 2] = -np.inf; a.tofile(self.o + '/A/doc1_b65536.f32'); self.err('non-finite.*row 1 col 2')
    def test_optional_arm_bad_file_rejected(self):
        mk(self.o, 'Ap', docs=(0,)); np.zeros(10, np.float32).tofile(self.o + '/Ap/doc0_b32768.f32'); self.err('Ap/doc0_b32768')
    def test_optional_arm_subset_ok(self): mk(self.o, 'Ap', docs=(0,), eps=0.3); A.main(self.o)
    def test_cli_exit_code(self):
        os.remove(self.o + '/B/doc0_b32768.f32')
        r = subprocess.run([sys.executable, '-I', os.path.join(D, 'g1c_analyze.py'), self.o], capture_output=True, text=True, env=dict(os.environ))
        self.assertEqual(r.returncode, 2); self.assertIn('ANALYZER ERROR: partial run', r.stderr)
if __name__ == '__main__': unittest.main(verbosity=2)
