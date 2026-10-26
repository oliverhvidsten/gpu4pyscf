/*
 * Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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
