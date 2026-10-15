#include "fixed_point.cuh"

extern "C" {
// 0 for a double build, otherwise the fixed-point shift
int fixed_point_shift(void)
{
#ifdef FIXED_POINT_ACCUM
    return FIXED_POINT_SHIFT;
#else
    return 0;
#endif
}
}
