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
Fixed-point (int64) accumulation for bit-reproducible GPU kernels.

Set GPU4PYSCF_FIXED_POINT=1 to load the lib*_fp.so variants built with
-DFIXED_POINT_ACCUM (see lib/fixed_point.cuh); the call sites of the accumulating
kernels then call from_fixed(buf) once after their kernel loop to convert the int64
accumulators back to double. The shift is read from the library.

Set GPU4PYSCF_DETERMINISTIC_BLOCKING=1 to make get_avail_mem() return a fixed
fraction of the device memory, so that block/batch sizes derived from it do not
depend on the momentary memory-pool state.

Either flag makes add_at() replace cupy.add.at (float atomics, order-dependent
rounding) by a one-hot matrix product, which cuBLAS evaluates in a fixed order.
'''

import os

ENABLED = os.environ.get('GPU4PYSCF_FIXED_POINT', '0').strip().lower() not in (
    '0', '', 'false', 'off', 'no')

DETERMINISTIC_BLOCKING = os.environ.get('GPU4PYSCF_DETERMINISTIC_BLOCKING', '0').strip().lower() not in (
    '0', '', 'false', 'off', 'no')
DETERMINISTIC_MEM_FRACTION = float(os.environ.get('GPU4PYSCF_DETERMINISTIC_MEM_FRACTION', '0.5'))


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


def deterministic():
    '''True when either determinism flag is set.'''
    return ENABLED or DETERMINISTIC_BLOCKING


def _onehot(labels, n_out, dtype):
    import cupy as cp
    labels = cp.asarray(labels).ravel()
    onehot = cp.zeros((n_out, labels.size), dtype=dtype)
    onehot[labels, cp.arange(labels.size)] = 1
    return onehot


def add_at(out, indices, values):
    '''out[indices] += values with duplicate indices accumulated, like
    cupy.add.at(out, indices, values).

    cupy.add.at scatters with floating-point atomics, so the summation order and
    hence the rounding differ from run to run. In deterministic mode the sum over
    each indexed axis is done as a matrix product with a one-hot (n_out x n)
    matrix instead, which is bit-reproducible and agrees with add.at to rounding.
    Otherwise this simply calls cupy.add.at.

    Supported indices (what the Hessian code uses): a 1-D integer array, or a
    tuple of slice(None) and integer arrays on consecutive axes, where the k
    arrays broadcast against each other as an outer product, e.g.
    (labels[:, None], labels[None, :], slice(None), slice(None)) or
    (labels[:, None], labels).
    `values` must have (or broadcast to) the shape of out[indices].
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
        # numpy broadcasting aligns shapes on the right: (n,) next to (n, 1) means (1, n)
        lab = lab.reshape((1,) * (k - lab.ndim) + lab.shape)
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
            # contract the indexed axis: (n_out, n) @ (..., n, L) -> (..., n_out, L)
            res = cp.moveaxis(cp.matmul(onehot, cp.moveaxis(res, ax, -2)), -2, ax)
    out += res
    return out
