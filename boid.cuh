#ifndef BOID_H
#define BOID_H

#include "floatmath.cuh"

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
};

__global__ void steerBoid(BoidSoA boids, float4* pos, float dt, int count, float separationWeight,
    float alignmentWeight, float cohesionWeight);

#endif