#ifndef H_DRAWING_H
#define H_DRAWING_H
#include "boid.cuh"

void createBoidsCPU(BoidSoA *boids);
void freeBoidsCPU(BoidSoA* boids);

#endif // H_DRAWING_H