# Bit-reproducible GPU calculations

This fork adds an optional deterministic mode to GPU4PySCF. With it, repeated
runs of the same input on the same GPU model give bit-identical SCF energies,
densities, gradients and analytic Hessians. Stock GPU4PySCF varies at the
1e-13 Ha / 1e-7 eV/Å / 1e-7 eV/Å² level between runs because its kernels reduce
with floating-point atomics, whose summation order depends on thread scheduling.

## Usage

```bash
export GPU4PYSCF_DETERMINISTIC=1
```

Set it before Python starts (the flag is read when `gpu4pyscf` is imported).
Nothing else changes: same API, same inputs. The two parts can also be enabled
separately with `GPU4PYSCF_FIXED_POINT=1` and `GPU4PYSCF_DETERMINISTIC_BLOCKING=1`.

## What it does

1. **Fixed-point kernels.** The CUDA libraries are built twice; the `lib*_fp.so`
   variants accumulate with int64 fixed-point atomics (48 fractional bits, the
   AMBER SPFP scheme) instead of double atomics, so the result does not depend on
   the order of the additions. The Python call sites convert the buffers back once.
2. **Deterministic blocking.** Block and batch sizes no longer depend on the
   momentary memory-pool state; the density-fitting tensor unpack writes each AO
   pair once; the Hessian code's `cupy.add.at` scatters are replaced by one-hot
   matrix products (`gpu4pyscf.lib.fixed_point.add_at`).

Measured cost: none (wall time ratio 0.99–1.05 for SCF, gradient and Hessian).
Deterministic and stock results agree to rounding (1e-13 Ha, 1e-9 Ha/Bohr²).

## Build

`pip install .` from this repository builds both library variants
(`-DBUILD_FIXED_POINT=ON`, the default; `-DFIXED_POINT_SHIFT=48`). The libraries
are linked with `-Bsymbolic`, which also fixes an intermittent illegal-address
crash in `int1e_grids` when the CUDA runtime is shared between the libraries.
A prebuilt wheel cannot come from PyPI: the stock wheels do not contain the
`_fp` libraries. Wheel build used for the group (Perlmutter, A100, CUDA 12.9;
remove a stale `build/` directory first, its CMake cache keeps old options):

```bash
module load cudatoolkit/12.9
export GPU4PYSCF_BUILD_CUDA_WHEEL=1 CUDACXX=$(which nvcc) CC=gcc CXX=g++
export CMAKE_CONFIGURE_ARGS="-DCUDA_ARCHITECTURES=80-real -DCMAKE_CUDA_RUNTIME_LIBRARY=Static -DBUILD_FIXED_POINT=ON"
pip wheel --no-deps --no-build-isolation -w dist .
```

## Limitations

- **Range.** A 48-bit-shift accumulator holds ±2^15. A single kernel term above
  that saturates and a sum beyond 2^63 wraps, both silently (no error, no NaN).
  Terms that grow with Z·ρ(0) reach this for third-row atoms: the nuclear-attraction
  Hessian is guarded by scaling the charges (`charge_scale_for_range`, calibrated on
  S and P with an 8× margin, extrapolated as Z⁴ beyond). Heavier elements and the
  other Hessian kernels have not been checked. Before trusting deterministic
  Hessians for a new element class, compare one against a run with the flag unset.
- `add_at` supports the index patterns used in `gpu4pyscf/hessian` and raises
  `NotImplementedError` for others.
- Validated on A100 for molecular DF-DFT with SMD solvent, including analytic
  Hessians, on molecules with H, C, N, O, F, P, S. Other code paths (PBC, TDDFT,
  direct-SCF Hessians) have the fixed-point kernels but were not validated.
- Bit-identity holds across runs and across nodes of the same GPU model. It is
  not expected across different GPU architectures or library versions.

## Tests

`pytest gpu4pyscf/lib/tests/test_fixed_point.py` checks `add_at` against
`cupy.add.at`, the range guard, and (when the `_fp` libraries are built)
bit-identity of SCF, gradient and Hessian between two fresh processes.
