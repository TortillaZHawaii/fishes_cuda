#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#include <vector_types.h>
#include <helper_math.h>

#include "boid.cuh"
#include "floatmath.cuh"

__device__ float3 bound_position(float3 pos)
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

__device__ float3 calculate_acceleration(float3 flockHeading, float3 centreOfMassSum, float3 avoidance, float3 position, int numPerceivedFlockmates)
{
    const float separationWeight = 5.0f;
    const float alignmentWeight = 5.0f;
    const float cohesionWeight = 1.0f;

    flockHeading /= numPerceivedFlockmates;

    centreOfMassSum /= numPerceivedFlockmates;
    centreOfMassSum -= position;

    float3 separation = limit3(avoidance, 1.f) * separationWeight;
    float3 alignment = limit3(flockHeading, 1.f) * alignmentWeight;
    float3 cohesion = limit3(centreOfMassSum, 1.f) * cohesionWeight;

    float3 acceleration = separation + alignment + cohesion;

    return acceleration;
}

__global__ void steerBoid(BoidSoA boids, float4* pos, float dt, int count) 
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= count)
    {
        return;
    }

    const float viewRadius = 0.1f;
    const float avoidRadius = 0.05f;

    const float maxSpeed = 1.0f;

    // sums
    float3 flockHeading = zero3(); // alignment
    float3 avoidance = zero3(); // separation
    float3 centreOfMassSum = zero3(); // cohesion

    int numPerceivedFlockmates = 0;

    // per boid
    float3 position = make_float3(boids.positionsX[tid], boids.positionsY[tid], boids.positionsZ[tid]);
    float3 velocity = make_float3(boids.velocitiesX[tid], boids.velocitiesY[tid], boids.velocitiesZ[tid]);
    float3 heading = make_float3(boids.headingsX[tid], boids.headingsY[tid], boids.headingsZ[tid]);

    // loop over all other boids
    for(int i = 0; i < count; ++i)
    {
        bool isTheSameBoid = i == tid;
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
        float3 acceleration = calculate_acceleration(flockHeading, centreOfMassSum, 
            avoidance, position, numPerceivedFlockmates);

        velocity += acceleration * dt;
        velocity = limit3(velocity, maxSpeed);
    }

    float3 inBoundAcceleration = bound_position(position);
    velocity += inBoundAcceleration * dt;

    // update position
    boids.positionsX[tid] += velocity.x * dt;
    boids.positionsY[tid] += velocity.y * dt;
    boids.positionsZ[tid] += velocity.z * dt;

    // update velocities
    boids.velocitiesX[tid] = velocity.x;
    boids.velocitiesY[tid] = velocity.y;
    boids.velocitiesZ[tid] = velocity.z;

    // update heading
    heading = normalize(velocity);

    boids.headingsX[tid] = heading.x;
    boids.headingsY[tid] = heading.y;
    boids.headingsZ[tid] = heading.z;

    position = make_float3(boids.positionsX[tid], boids.positionsY[tid], boids.positionsZ[tid]);

    // write output vertex
    // head
    pos[2 * tid] = make_float4(position.x, position.y, position.z, 1.0f);
    // tail
    float tailSize = 0.01f;
    float3 tailPosition = position - heading * tailSize;
    pos[2 * tid + 1] = make_float4(tailPosition.x, tailPosition.y, tailPosition.z, 1.0f);
}
