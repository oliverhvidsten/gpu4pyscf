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

'''
Bit-reproducible GPU reductions. GPU4PYSCF_DETERMINISTIC=1 enables both parts;
each can also be set on its own:

  GPU4PYSCF_FIXED_POINT=1             load the lib*_fp.so variants, whose kernels
                                      accumulate in int64 fixed point (fixed_point.cuh);
                                      their call sites convert back with from_fixed()
  GPU4PYSCF_DETERMINISTIC_BLOCKING=1  block sizes independent of the memory-pool state,
                                      first-occurrence writes when unpacking cderi, and
                                      add_at() instead of cupy.add.at

Range: with shift 48 an accumulator holds +-2^15; larger terms saturate or wrap
silently. charge_scale_for_range() guards the one known case. See DETERMINISM.md.
'''

import math
import os


def _flag(name, default=False):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ('0', '', 'false', 'off', 'no')


_all = _flag('GPU4PYSCF_DETERMINISTIC')
ENABLED = _flag('GPU4PYSCF_FIXED_POINT', _all)
DETERMINISTIC_BLOCKING = _flag('GPU4PYSCF_DETERMINISTIC_BLOCKING', _all)
DETERMINISTIC_MEM_FRACTION = float(os.environ.get('GPU4PYSCF_DETERMINISTIC_MEM_FRACTION', '0.5'))


def deterministic():
    '''True when either part is on.'''
    return ENABLED or DETERMINISTIC_BLOCKING


def deterministic_avail_mem():
    import cupy
    total = cupy.cuda.runtime.memGetInfo()[1]
    return int(total * DETERMINISTIC_MEM_FRACTION)


# libraries that have a fixed-point build and whose call sites call from_fixed()
FP_LIBRARIES = ('libgvhf_rys', 'libgvhf_md', 'libgint', 'libpbc')

_shift = None


def library_name(libname):
    '''File stem to load for `libname`: lib<name>_fp when fixed point is enabled.'''
    if ENABLED and libname in FP_LIBRARIES:
        return libname + '_fp'
    return libname


def register(libname, lib):
    '''Record (and cross-check) the shift of a loaded fixed-point library.'''
    global _shift
    if not (ENABLED and libname in FP_LIBRARIES):
        return
    try:
        shift = int(lib.fixed_point_shift())
    except AttributeError:
        raise RuntimeError(f'{libname}_fp does not export fixed_point_shift(); '
                           'it was not built with -DFIXED_POINT_ACCUM')
    if shift <= 0:
        raise RuntimeError(f'{libname}_fp was built without fixed-point accumulation')
    if _shift is None:
        _shift = shift
    elif shift != _shift:
        raise RuntimeError(f'fixed-point shift mismatch: {libname}_fp uses {shift}, '
                           f'another library uses {_shift}')


def shift():
    '''The fixed-point shift of the loaded libraries (None when disabled).'''
    return _shift


def from_fixed(buf):
    '''Convert, in place, a zero-initialised float64 buffer that the fixed-point
    kernels accumulated into. No-op (returns buf) when fixed point is disabled.'''
    if not ENABLED or buf is None:
        return buf
    import cupy as cp
    if _shift is None:
        raise RuntimeError('from_fixed() called but no fixed-point library is loaded')
    assert buf.dtype == cp.float64 and buf.flags.c_contiguous
    buf[...] = buf.view(cp.int64).astype(cp.float64) * (2.0 ** -_shift)
    return buf


def _onehot(labels, n_out, dtype):
    import cupy as cp
    labels = cp.asarray(labels).ravel()
    onehot = cp.zeros((n_out, labels.size), dtype=dtype)
    onehot[labels, cp.arange(labels.size)] = 1
    return onehot


def add_at(out, indices, values):
    '''cupy.add.at(out, indices, values) without float atomics: in deterministic
    mode each indexed axis is summed by a one-hot matrix product (fixed order, so
    bit-reproducible; agrees with add.at to rounding), otherwise plain cupy.add.at.

    indices: a 1-D integer array, or a tuple of slice(None) and integer arrays on
    consecutive axes that broadcast as an outer product, e.g. (lab[:, None], lab)
    or (lab[:, None], lab[None, :], slice(None), slice(None)).
    '''
    import cupy as cp
    if not deterministic():
        cp.add.at(out, indices, values)
        return out

    if not isinstance(indices, tuple):
        indices = (indices,)
    axes, labels = [], []
    for ax, idx in enumerate(indices):
        if isinstance(idx, slice):
            if idx != slice(None):
                raise NotImplementedError('add_at: only slice(None) is supported')
            continue
        axes.append(ax)
        labels.append(cp.asarray(idx))
    k = len(axes)
    if k == 0:
        raise ValueError('add_at: no index array given')
    if axes != list(range(axes[0], axes[0] + k)):
        raise NotImplementedError('add_at: index arrays must be on consecutive axes')
    for j, lab in enumerate(labels):
        if lab.ndim > k:
            raise NotImplementedError('add_at: index array has more dimensions than '
                                      'there are index arrays')
        lab = lab.reshape((1,) * (k - lab.ndim) + lab.shape)  # numpy right-aligns shapes
        wide = [d for d, n in enumerate(lab.shape) if n != 1]
        if len(wide) > 1 or (wide and wide[0] != j):
            raise NotImplementedError('add_at: index arrays must form an outer product')
        labels[j] = lab.ravel()

    values = cp.asarray(values)
    expected = list(out.shape)
    expected[axes[0]:axes[0] + k] = [lab.size for lab in labels]
    if list(values.shape) != expected:
        values = cp.broadcast_to(values, expected)

    res = values
    for ax, lab in zip(axes, labels):
        onehot = _onehot(lab, out.shape[ax], out.dtype)
        if res.ndim == 1:
            res = onehot @ res
        else:
            # (n_out, n) @ (..., n, L) -> (..., n_out, L)
            res = cp.moveaxis(cp.matmul(onehot, cp.moveaxis(res, ax, -2)), -2, ax)
    out += res
    return out


def charge_scale_for_range(max_charge):
    '''Power of two by which to divide point charges fed to a fixed-point kernel
    whose terms grow like Z*rho(0) ~ Z^4, so they stay inside the accumulator
    range: 1 for Z <= 8, 16 for P/S, 32 for Cl. Calibrated on S, extrapolated
    beyond. 1.0 when fixed point is off.'''
    if not ENABLED or max_charge <= 8:
        return 1.0
    return 2.0 ** math.ceil(4 * math.log2(max_charge / 8.0))
