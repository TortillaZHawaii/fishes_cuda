#include "floatmath.cuh"
#include <helper_math.h>

__device__ __host__ float3 zero3()
{
    return make_float3(0.0f);
}

__device__ __host__ float3 limit3(const float3 a, const float max)
{
    float length = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
    if (length > max) {
        float newLength = max / length;
        return a * newLength;
    }
    return a;
}

// generates random value in range [min, max)
float randFloatInRange(float min, float max)
{
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = max - min;
    float r = random * diff;
    return min + r;
}
