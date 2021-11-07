#include "floatmath.cuh"
#include <helper_math.h>

__device__ float3 zero3()
{
    return make_float3(0.0f, 0.0f, 0.0f);
}

__device__ float3 limit3(const float3 a, const float max)
{
    float length = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
    if (length > max) {
        float newLength = max / length;
        return a * newLength;
    }
    return a;
}
