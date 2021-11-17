#include "d_drawing.cuh"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <vector_types.h>
#include <helper_math.h>
#include "defines.cuh"
#include "floatmath.cuh"

void randomizeBoids(BoidSoA* boids);

// allocates memory for boids on GPU and initializes them with random values
void createBoids(BoidSoA *boids)
{
    checkCudaErrors(cudaMalloc(&boids->headingsX, sizeof(float) * BOID_COUNT));
    checkCudaErrors(cudaMalloc(&boids->headingsY, sizeof(float) * BOID_COUNT));
    checkCudaErrors(cudaMalloc(&boids->headingsZ, sizeof(float) * BOID_COUNT));
    
    checkCudaErrors(cudaMalloc(&boids->positionsX, sizeof(float) * BOID_COUNT));
    checkCudaErrors(cudaMalloc(&boids->positionsY, sizeof(float) * BOID_COUNT));
    checkCudaErrors(cudaMalloc(&boids->positionsZ, sizeof(float) * BOID_COUNT));

    checkCudaErrors(cudaMemset(boids->headingsX, 0, sizeof(float) * BOID_COUNT));
    checkCudaErrors(cudaMemset(boids->headingsY, 0, sizeof(float) * BOID_COUNT));
    checkCudaErrors(cudaMemset(boids->headingsZ, 0, sizeof(float) * BOID_COUNT));

    checkCudaErrors(cudaMemset(boids->positionsX, 0, sizeof(float) * BOID_COUNT));
    checkCudaErrors(cudaMemset(boids->positionsY, 0, sizeof(float) * BOID_COUNT));
    checkCudaErrors(cudaMemset(boids->positionsZ, 0, sizeof(float) * BOID_COUNT));

    checkCudaErrors(cudaMalloc(&boids->velocitiesX, sizeof(float) * BOID_COUNT));
    checkCudaErrors(cudaMalloc(&boids->velocitiesY, sizeof(float) * BOID_COUNT));
    checkCudaErrors(cudaMalloc(&boids->velocitiesZ, sizeof(float) * BOID_COUNT));

    randomizeBoids(boids);
}

// generates random values for boids on CPU and copies them to GPU
void randomizeBoids(BoidSoA* boids)
{
    BoidSoA h_boids;

    h_boids.positionsX = (float*)malloc(sizeof(float) * BOID_COUNT);
    h_boids.positionsY = (float*)malloc(sizeof(float) * BOID_COUNT);
    h_boids.positionsZ = (float*)malloc(sizeof(float) * BOID_COUNT);

    h_boids.velocitiesX = (float*)malloc(sizeof(float) * BOID_COUNT);
    h_boids.velocitiesY = (float*)malloc(sizeof(float) * BOID_COUNT);
    h_boids.velocitiesZ = (float*)malloc(sizeof(float) * BOID_COUNT);

    h_boids.headingsX = (float*)malloc(sizeof(float) * BOID_COUNT);
    h_boids.headingsY = (float*)malloc(sizeof(float) * BOID_COUNT);
    h_boids.headingsZ = (float*)malloc(sizeof(float) * BOID_COUNT);

    if(h_boids.positionsX == NULL || h_boids.positionsY == NULL || h_boids.positionsZ == NULL ||
       h_boids.velocitiesX == NULL || h_boids.velocitiesY == NULL || h_boids.velocitiesZ == NULL ||
       h_boids.headingsX == NULL || h_boids.headingsY == NULL || h_boids.headingsZ == NULL)
    {
        printf("Error allocating memory for boids\n");
        exit(EXIT_FAILURE);
    }

    const float max_velocity = 0.2f;

    for(int i = 0; i < BOID_COUNT; i++)
    {
        h_boids.positionsX[i] = randFloatInRange(-1.0f, 1.0f);
        h_boids.positionsY[i] = randFloatInRange(-1.0f, 1.0f);
        h_boids.positionsZ[i] = randFloatInRange(-1.0f, 1.0f);

        h_boids.velocitiesX[i] = randFloatInRange(-max_velocity, max_velocity);
        h_boids.velocitiesY[i] = randFloatInRange(-max_velocity, max_velocity);
        h_boids.velocitiesZ[i] = randFloatInRange(-max_velocity, max_velocity);

        float3 heading = make_float3(h_boids.velocitiesX[i], h_boids.velocitiesY[i], h_boids.velocitiesZ[i]);
        heading = normalize(heading);

        h_boids.headingsX[i] = heading.x;
        h_boids.headingsY[i] = heading.y;
        h_boids.headingsZ[i] = heading.z;
    }

    checkCudaErrors(cudaMemcpy(boids->positionsX, h_boids.positionsX, sizeof(float) * BOID_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(boids->positionsY, h_boids.positionsY, sizeof(float) * BOID_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(boids->positionsZ, h_boids.positionsZ, sizeof(float) * BOID_COUNT, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(boids->velocitiesX, h_boids.velocitiesX, sizeof(float) * BOID_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(boids->velocitiesY, h_boids.velocitiesY, sizeof(float) * BOID_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(boids->velocitiesZ, h_boids.velocitiesZ, sizeof(float) * BOID_COUNT, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(boids->headingsX, h_boids.headingsX, sizeof(float) * BOID_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(boids->headingsY, h_boids.headingsY, sizeof(float) * BOID_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(boids->headingsZ, h_boids.headingsZ, sizeof(float) * BOID_COUNT, cudaMemcpyHostToDevice));

    free(h_boids.positionsX);
    free(h_boids.positionsY);
    free(h_boids.positionsZ);

    free(h_boids.velocitiesX);
    free(h_boids.velocitiesY);
    free(h_boids.velocitiesZ);

    free(h_boids.headingsX);
    free(h_boids.headingsY);
    free(h_boids.headingsZ);
}

// frees GPU memory
void freeBoids(BoidSoA* boids)
{
    checkCudaErrors(cudaFree(boids->headingsX));
    checkCudaErrors(cudaFree(boids->headingsY));
    checkCudaErrors(cudaFree(boids->headingsZ));

    checkCudaErrors(cudaFree(boids->positionsX));
    checkCudaErrors(cudaFree(boids->positionsY));
    checkCudaErrors(cudaFree(boids->positionsZ));

    checkCudaErrors(cudaFree(boids->velocitiesX));
    checkCudaErrors(cudaFree(boids->velocitiesY));
    checkCudaErrors(cudaFree(boids->velocitiesZ));
}