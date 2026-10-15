/*
 * Order-independent (bit-reproducible) accumulation with -DFIXED_POINT_ACCUM:
 * fp_atomic_add(double*, double) scales the value by 2^FIXED_POINT_SHIFT, rounds
 * to int64 and adds it with an integer atomicAdd into the same 8-byte slot
 * (AMBER SPFP scheme). The buffer must start zero-filled and is converted back
 * once with gpu4pyscf.lib.fixed_point.from_fixed(). Without the define this is
 * the plain double atomicAdd. Default shift 48: resolution 2^-48, range +-2^15.
 */
#pragma once

#ifdef __CUDACC__

#ifdef FIXED_POINT_ACCUM
#ifndef FIXED_POINT_SHIFT
#define FIXED_POINT_SHIFT 48
#endif
#define FIXED_POINT_SCALE ((double)(1ULL << FIXED_POINT_SHIFT))

__device__ __forceinline__ void fp_atomic_add(double *addr, double val)
{
    long long iv = __double2ll_rn(val * FIXED_POINT_SCALE);
    atomicAdd(reinterpret_cast<unsigned long long*>(addr),
              static_cast<unsigned long long>(iv));
}
#else
__device__ __forceinline__ void fp_atomic_add(double *addr, double val)
{
    atomicAdd(addr, val);
}
#endif

// pass-through overloads for the integer task counters etc.
__device__ __forceinline__ int fp_atomic_add(int *addr, int val)
{ return atomicAdd(addr, val); }
__device__ __forceinline__ unsigned int fp_atomic_add(unsigned int *addr, unsigned int val)
{ return atomicAdd(addr, val); }
__device__ __forceinline__ unsigned long long fp_atomic_add(unsigned long long *addr, unsigned long long val)
{ return atomicAdd(addr, val); }
__device__ __forceinline__ float fp_atomic_add(float *addr, float val)
{ return atomicAdd(addr, val); }

#endif // __CUDACC__
