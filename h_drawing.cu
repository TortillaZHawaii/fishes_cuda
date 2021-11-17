#include "h_drawing.cuh"

#include <vector_types.h>
#include <helper_math.h>
#include "defines.cuh"

void randomizeBoids(BoidSoA* boids);

// allocates memory for boids on GPU and initializes them with random values
void createBoids(BoidSoA *boids)
{
    boids->positionsX = (float*)malloc(sizeof(float) * BOID_COUNT);
    boids->positionsY = (float*)malloc(sizeof(float) * BOID_COUNT);
    boids->positionsZ = (float*)malloc(sizeof(float) * BOID_COUNT);

    boids->velocitiesX = (float*)malloc(sizeof(float) * BOID_COUNT);
    boids->velocitiesY = (float*)malloc(sizeof(float) * BOID_COUNT);
    boids->velocitiesZ = (float*)malloc(sizeof(float) * BOID_COUNT);

    boids->headingsX = (float*)malloc(sizeof(float) * BOID_COUNT);
    boids->headingsY = (float*)malloc(sizeof(float) * BOID_COUNT);
    boids->headingsZ = (float*)malloc(sizeof(float) * BOID_COUNT);

    boids->masses = (float*)malloc(sizeof(float) * BOID_COUNT);

    if(boids->positionsX == NULL || boids->positionsY == NULL || boids->positionsZ == NULL ||
       boids->velocitiesX == NULL || boids->velocitiesY == NULL || boids->velocitiesZ == NULL ||
       boids->headingsX == NULL || boids->headingsY == NULL || boids->headingsZ == NULL || 
       boids->masses == NULL)
    {
        printf("Error allocating memory for boids\n");
        exit(EXIT_FAILURE);
    }

    randomizeBoids(boids);
}

// generates random values for boids on CPU and copies them to GPU
void randomizeBoids(BoidSoA* boids)
{
    const float max_velocity = 0.2f;
    const float min_mass = 0.5f;
    const float max_mass = 5.0f;

    for(int i = 0; i < BOID_COUNT; i++)
    {
        boids->positionsX[i] = randFloatInRange(-1.0f, 1.0f);
        boids->positionsY[i] = randFloatInRange(-1.0f, 1.0f);
        boids->positionsZ[i] = randFloatInRange(-1.0f, 1.0f);

        boids->velocitiesX[i] = randFloatInRange(-max_velocity, max_velocity);
        boids->velocitiesY[i] = randFloatInRange(-max_velocity, max_velocity);
        boids->velocitiesZ[i] = randFloatInRange(-max_velocity, max_velocity);

        boids->masses[i] = randFloatInRange(min_mass, max_mass);

        float3 heading = make_float3(boids->velocitiesX[i], boids->velocitiesY[i], boids->velocitiesZ[i]);
        heading = normalize(heading);

        boids->headingsX[i] = heading.x;
        boids->headingsY[i] = heading.y;
        boids->headingsZ[i] = heading.z;
    }
}

// frees GPU memory
void freeBoids(BoidSoA* boids)
{
    free(boids->positionsX);
    free(boids->positionsY);
    free(boids->positionsZ);

    free(boids->velocitiesX);
    free(boids->velocitiesY);
    free(boids->velocitiesZ);

    free(boids->headingsX);
    free(boids->headingsY);
    free(boids->headingsZ);

    free(boids->masses);
}