#ifndef D_DRAWING_H
#define D_DRAWING_H
#include "boid.cuh"

void createBoids(BoidSoA *boids);
void freeBoids(BoidSoA* boids);

#endif // D_DRAWING_H