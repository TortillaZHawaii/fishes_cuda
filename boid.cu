#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#include <vector_types.h>
#include <helper_math.h>

#include "boid.cuh"
#include "floatmath.cuh"

__device__ __host__ float3 bound_position(float3 pos)
{
    float3 acc = zero3();

    const float goBackAcc = 4.0f;
    const float min = -0.95f;
    const float max = -min;

    if (pos.x < min)
    {
        acc.x = goBackAcc;
    }
    else if (pos.x > max)
    {
        acc.x = -goBackAcc;
    }

    if(pos.y < min)
    {
        acc.y = goBackAcc;
    }
    else if(pos.y > max)
    {
        acc.y = -goBackAcc;
    }

    if(pos.z < min)
    {
        acc.z = goBackAcc;
    }
    else if(pos.z > max)
    {
        acc.z = -goBackAcc;
    }

    return acc;
}

__device__ __host__ float3 calculateForce(float3 flockHeading, float3 centreOfMassSum, float3 avoidance, 
    float3 position, int numPerceivedFlockmates, float separationWeight, float alignmentWeight, 
    float cohesionWeight)
{
    flockHeading /= numPerceivedFlockmates;

    centreOfMassSum /= numPerceivedFlockmates;
    centreOfMassSum -= position;

    float3 separation = limit3(avoidance, 1.f) * separationWeight;
    float3 alignment = limit3(flockHeading, 1.f) * alignmentWeight;
    float3 cohesion = limit3(centreOfMassSum, 1.f) * cohesionWeight;

    float3 force = separation + alignment + cohesion;

    return force;
}

__device__ __host__ void updatePositionInSoA(BoidSoA boids, int tid, float3 offset)
{
    boids.positionsX[tid] += offset.x;
    boids.positionsY[tid] += offset.y;
    boids.positionsZ[tid] += offset.z;
}

__device__ __host__ void updateVelocityInSoA(BoidSoA boids, int tid, float3 velocity)
{
    boids.velocitiesX[tid] = velocity.x;
    boids.velocitiesY[tid] = velocity.y;
    boids.velocitiesZ[tid] = velocity.z;
}

__device__ __host__ void updateHeadingInSoA(BoidSoA boids, int tid, float3 velocity)
{
    float3 heading = normalize(velocity);

    boids.headingsX[tid] = heading.x;
    boids.headingsY[tid] = heading.y;
    boids.headingsZ[tid] = heading.z;
}

// adapted from https://www.freemancw.com/2012/06/opengl-cone-function/
__device__ __host__ float3 perp(float3 v)
{
    float min = fabsf(v.x);
    float3 cardinalAxis = make_float3(1.0f, 0.0f, 0.0f);

    if (fabsf(v.y) < min)
    {
        min = fabsf(v.y);
        cardinalAxis = make_float3(0.0f, 1.0f, 0.0f);
    }

    if (fabsf(v.z) < min)
    {
        cardinalAxis = make_float3(0.0f, 0.0f, 1.0f);
    }

    return cross(v, cardinalAxis);
}

// adapted from https://www.freemancw.com/2012/06/opengl-cone-function/
__device__ __host__ void updatePyramidPosition(float4* pos, BoidSoA boids, int tid)
{
    float3 position = make_float3(boids.positionsX[tid], boids.positionsY[tid], boids.positionsZ[tid]);
    float3 heading = make_float3(boids.headingsX[tid], boids.headingsY[tid], boids.headingsZ[tid]);

    float3 e0 = perp(heading);
    float3 e1 = cross(e0, heading);
    
    const float angInc = 2 * M_PI / BOTTOM_VERTICES_COUNT;

    int startId = tid * VERTICES_TO_DRAW_PER_BOID;

    // head
    const float length = 0.015f;
    float3 headPosition = position + heading * length * sqrtf(boids.masses[tid]);
    float4 bottomPoints4[BOTTOM_VERTICES_COUNT];

    // tail
    const float tailSize = 0.008f;
    for(int i = 0; i < BOTTOM_VERTICES_COUNT; ++i) // one extra to close the pyramid
    {
        float ang = i * angInc;
        float3 bottomPoint = position + (e0 * cosf(ang) + e1 * sinf(ang)) * tailSize;
        bottomPoints4[i] = make_float4(bottomPoint.x, bottomPoint.y, bottomPoint.z, 1.0f);
    }

    // write output vertices
    // base triangle
    for(int i = 0; i < BOTTOM_VERTICES_COUNT; ++i)
    {
        pos[startId + i] = bottomPoints4[i];
    }

    float4 head4 = make_float4(headPosition.x, headPosition.y, headPosition.z, 1.0f);
    // upper triangles
    for(int i = 0; i < BOTTOM_VERTICES_COUNT; ++i)
    {
        pos[startId + (i + 1) * TRIANGLE_VERTICES_COUNT] = head4;
        pos[startId + (i + 1) * TRIANGLE_VERTICES_COUNT + 1] = bottomPoints4[i];
        pos[startId + (i + 1) * TRIANGLE_VERTICES_COUNT + 2] = bottomPoints4[(i + 1) % BOTTOM_VERTICES_COUNT];
    }
}

__device__ __host__ void steerBoid(int id, BoidSoA boids, float4* linePos, float dt, int count, float separationWeight,
    float alignmentWeight, float cohesionWeight)
{
    const float viewRadius = 0.1f;
    const float avoidRadius = 0.05f;

    const float maxSpeed = 1.0f;

    // sums
    float3 flockHeading = zero3(); // alignment
    float3 avoidance = zero3(); // separation
    float3 centreOfMassSum = zero3(); // cohesion

    int numPerceivedFlockmates = 0;

    // per boid
    float3 position = make_float3(boids.positionsX[id], boids.positionsY[id], boids.positionsZ[id]);
    float3 velocity = make_float3(boids.velocitiesX[id], boids.velocitiesY[id], boids.velocitiesZ[id]);
    float3 heading = make_float3(boids.headingsX[id], boids.headingsY[id], boids.headingsZ[id]);
    float mass = boids.masses[id];

    // loop over all other boids
    for(int i = 0; i < count; ++i)
    {
        bool isTheSameBoid = i == id;
        if(isTheSameBoid)
        {
            continue;
        }

        float3 otherPos = make_float3(boids.positionsX[i], boids.positionsY[i], boids.positionsZ[i]);

        float3 offset = otherPos - position;

        float sqrDistance = dot(offset, offset);

        bool isInRange = sqrDistance < viewRadius * viewRadius;
        if(!isInRange)
        {
            continue;
        }

        // in range boid other than ourself
        ++numPerceivedFlockmates;

        float3 otherHeading = make_float3(boids.headingsX[i], boids.headingsY[i], boids.headingsZ[i]);
        flockHeading += otherHeading;
        centreOfMassSum += otherPos;

        bool isInAvoidRange = sqrDistance < avoidRadius * avoidRadius;
        if(isInAvoidRange)
        {
            avoidance -= offset;
        }
    }

    // adjust position and heading
    bool hasNeighbors = numPerceivedFlockmates > 0;
    if(hasNeighbors)
    {
        float3 force = calculateForce(flockHeading, centreOfMassSum, 
            avoidance, position, numPerceivedFlockmates, separationWeight, alignmentWeight,
            cohesionWeight);
        
        float3 acceleration = force * mass;

        velocity += acceleration * dt;
        velocity = limit3(velocity, maxSpeed);
    }

    // keep in bounds force
    float3 inBoundForce = bound_position(position);
    velocity += inBoundForce / mass * dt;

    updatePositionInSoA(boids, id, velocity * dt);
    updateVelocityInSoA(boids, id, velocity);
    updateHeadingInSoA(boids, id, velocity);

    updatePyramidPosition(linePos, boids, id);
}

__global__ void d_steerBoid(BoidSoA boids, float4* linePos, float dt, int count, float separationWeight,
    float alignmentWeight, float cohesionWeight) 
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= count)
    {
        return;
    }

    steerBoid(tid, boids, linePos, dt, count, separationWeight, alignmentWeight, cohesionWeight);
}

void h_steerBoid(BoidSoA boids, float4* pos, float dt, int count, float separationWeight,
    float alignmentWeight, float cohesionWeight)
{
    for(int i = 0; i < count; ++i)
    {
        steerBoid(i, boids, pos, dt, count, separationWeight, alignmentWeight, cohesionWeight);
    }
}
