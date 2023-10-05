# Shoal of fish - CUDA

Simple simulation of boid behaviour in 3D space. Implemented using CUDA and OpenGL.


https://github.com/TortillaZHawaii/fishes_cuda/assets/62249621/a98f24a2-8339-4358-80b0-bd5ef60b411c


## Installation
1. Install CUDA with  https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html, as well as third party libraries https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#install-third-party-libraries.
2. Using `git` get https://github.com/NVIDIA/cuda-samples.
3. In `Makefile` set `CU_LIBS` to point to `cuda-samples/Common`. Example `CU_LIBS=-I"~/cuda-samples/Common"`.
4. Run `make all` and you are set to run `fishes`.

## Usage
User can change parameters of simulation:
1. Cohesion - press 1 or 2.
2. Separation - press 3 or 4.
3. Alignment - press 5 or 6.
4. CPU/GPU mode - uncomment `// #define CPU` in `main.cu` and recompile (`make clean && make all`) to run on CPU. Leave commented out to run on GPU.
5. Number of boids - update `BOID_COUNT` in `defines.cuh` and recompile (`make clean && make all`).
