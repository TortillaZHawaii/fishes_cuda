#ifndef FLOATMATH_H
#define FLOATMATH_H

__device__ __host__ float3 zero3();

__device__ __host__ float3 limit3(const float3 a, const float max);

// generates random value in range [min, max)
float randFloatInRange(float min, float max);

#endif