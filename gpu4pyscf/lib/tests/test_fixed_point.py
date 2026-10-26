# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import json
import subprocess
import unittest
import numpy as np
import cupy as cp
from gpu4pyscf.lib import fixed_point
from gpu4pyscf.lib.fixed_point import add_at, charge_scale_for_range

LIBDIR = os.path.join(os.path.dirname(fixed_point.__file__))
HAVE_FP_LIBS = all(os.path.exists(os.path.join(LIBDIR, f'{lib}_fp.so'))
                   for lib in fixed_point.FP_LIBRARIES)

# One SCF + gradient + Hessian in a fresh process (the flags are read at import).
_CHILD = r'''
import sys, json, numpy as np, pyscf
from gpu4pyscf import dft
mol = pyscf.M(atom="O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59; F 1.8 0 0.2; S 1.2 1.5 -0.4",
              basis="def2-svp", charge=0, spin=1, verbose=0)
mf = dft.UKS(mol, xc="wb97x").density_fit()
mf.grids.level = 1
mf.conv_tol = 1e-10
mf.kernel()
g = mf.nuc_grad_method().kernel()
h = mf.Hessian().kernel()
np.savez(sys.argv[1], e=mf.e_tot, g=g, h=h)
'''


def _run_child(out, env):
    e = dict(os.environ)
    e.pop('GPU4PYSCF_DETERMINISTIC', None); e.pop('GPU4PYSCF_FIXED_POINT', None)
    e.pop('GPU4PYSCF_DETERMINISTIC_BLOCKING', None)
    e.update(env)
    subprocess.run([sys.executable, '-c', _CHILD, out], check=True, env=e)
    return np.load(out)


class KnownValues(unittest.TestCase):
    def setUp(self):
        self._saved = (fixed_point.ENABLED, fixed_point.DETERMINISTIC_BLOCKING)
        fixed_point.DETERMINISTIC_BLOCKING = True   # add_at takes the one-hot path

    def tearDown(self):
        fixed_point.ENABLED, fixed_point.DETERMINISTIC_BLOCKING = self._saved

    def _check(self, shape, idx, vals):
        ref = cp.zeros(shape); cp.add.at(ref, idx, vals)
        got = cp.zeros(shape); add_at(got, idx, vals)
        scale = max(float(cp.abs(ref).max()), 1.0)
        self.assertLess(float(cp.abs(ref - got).max()), 1e-13 * scale)
        again = add_at(cp.zeros(shape), idx, vals)
        self.assertTrue(cp.array_equal(got, again))

    def test_add_at_patterns(self):
        rng = np.random.default_rng(1)
        natm, nao, naux = 7, 61, 97
        lab = rng.integers(0, natm, nao)
        aux = rng.integers(0, natm, naux)
        rnd = lambda *s: cp.asarray(rng.standard_normal(s) * 10 ** rng.uniform(-6, 2, s))
        self._check((natm,), lab, rnd(nao))
        self._check((natm, 3, 3), lab, rnd(nao, 3, 3))
        self._check((natm, nao, 3, 3), lab, rnd(nao, nao, 3, 3))
        self._check((nao, natm, 3, 3), (slice(None), lab), rnd(nao, nao, 3, 3))
        self._check((natm, natm, 3, 3), (lab[:, None], lab), rnd(nao, nao, 3, 3))
        self._check((natm, natm, 3, 3), (aux[:, None], aux), rnd(naux, naux, 3, 3))
        self._check((natm, natm, 3, 3),
                    (cp.asarray(lab)[:, None], cp.asarray(lab)[None, :], slice(None), slice(None)),
                    rnd(nao, nao, 3, 3))

    def test_add_at_view_output(self):
        rng = np.random.default_rng(2)
        natm, nao, ng = 5, 40, 300
        lab = cp.asarray(rng.integers(0, natm, nao))
        v = cp.asarray(rng.standard_normal((3, nao, ng)))
        ref = cp.zeros((natm, 3, 5, ng)); out = cp.zeros_like(ref)
        cp.add.at(ref[:, :, 0, :], lab, v.transpose(1, 0, 2))
        add_at(out[:, :, 0, :], lab, v.transpose(1, 0, 2))
        self.assertLess(float(cp.abs(ref - out).max()), 1e-12)
        self.assertTrue(bool((out[:, :, 1:, :] == 0).all()))

    def test_add_at_rejects_unsupported(self):
        lab = cp.arange(4)
        for bad in [(lab, slice(None), lab), (slice(0, 2), lab), (lab[:, None], lab[:, None])]:
            with self.assertRaises((NotImplementedError, ValueError)):
                add_at(cp.zeros((4, 4, 4)), bad, cp.zeros((4, 4, 4)))

    def test_add_at_fallback(self):
        fixed_point.DETERMINISTIC_BLOCKING = False
        fixed_point.ENABLED = False
        out = cp.zeros(3)
        add_at(out, cp.asarray([0, 0, 2]), cp.ones(3))
        self.assertEqual(out.tolist(), [2.0, 0.0, 1.0])

    def test_charge_scale(self):
        fixed_point.ENABLED = True
        self.assertEqual(charge_scale_for_range(1), 1.0)
        self.assertEqual(charge_scale_for_range(8), 1.0)
        self.assertEqual(charge_scale_for_range(9), 2.0)
        self.assertEqual(charge_scale_for_range(15), 16.0)
        self.assertEqual(charge_scale_for_range(16), 16.0)
        self.assertEqual(charge_scale_for_range(17), 32.0)
        fixed_point.ENABLED = False
        self.assertEqual(charge_scale_for_range(16), 1.0)

    def test_from_fixed_disabled_is_noop(self):
        fixed_point.ENABLED = False
        buf = cp.ones(3)
        self.assertIs(fixed_point.from_fixed(buf), buf)

    @unittest.skipUnless(HAVE_FP_LIBS, 'fixed-point libraries not built (BUILD_FIXED_POINT=OFF)')
    def test_bit_identical_scf_grad_hess(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            det1 = _run_child(os.path.join(d, 'd1.npz'), {'GPU4PYSCF_DETERMINISTIC': '1'})
            det2 = _run_child(os.path.join(d, 'd2.npz'), {'GPU4PYSCF_DETERMINISTIC': '1'})
            ref = _run_child(os.path.join(d, 'ref.npz'), {})
        for key in ('e', 'g', 'h'):
            self.assertTrue(np.array_equal(det1[key], det2[key]), f'{key} not bit-identical')
        self.assertLess(abs(float(det1['e']) - float(ref['e'])), 1e-10)
        self.assertLess(np.abs(det1['g'] - ref['g']).max(), 1e-8)
        self.assertLess(np.abs(det1['h'] - ref['h']).max(), 1e-7)   # S present: exercises the range guard


if __name__ == "__main__":
    print("Full tests for fixed_point")
    unittest.main()
