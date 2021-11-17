#ifndef BOID_H
#define BOID_H

#include "floatmath.cuh"

#define TRIANGLE_VERTICES_COUNT 3
#define BOTTOM_VERTICES_COUNT TRIANGLE_VERTICES_COUNT // must be a triangle
#define NUMBER_OF_TRIANGLES (BOTTOM_VERTICES_COUNT + 1) // +1 for base
#define VERTICES_TO_DRAW_PER_BOID (NUMBER_OF_TRIANGLES * 3)

struct BoidSoA {
    float* positionsX;
    float* positionsY;
    float* positionsZ;

    // normalized velocity
    float* headingsX;
    float* headingsY;
    float* headingsZ;

    float* velocitiesX;
    float* velocitiesY;
    float* velocitiesZ;

    float* masses;
};

__device__ __host__ void steerBoid(int id, BoidSoA boids, float4* linePos, float dt, int count, 
    float separationWeight, float alignmentWeight, float cohesionWeight);

__global__ void d_steerBoid(BoidSoA boids, float4* pos, float dt, int count, float separationWeight,
    float alignmentWeight, float cohesionWeight);

void h_steerBoid(BoidSoA boids, float4* pos, float dt, int count, float separationWeight,
    float alignmentWeight, float cohesionWeight);

#endif