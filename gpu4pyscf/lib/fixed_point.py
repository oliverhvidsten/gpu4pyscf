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
