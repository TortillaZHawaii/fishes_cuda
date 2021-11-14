/* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined (__APPLE__) || defined(MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#include <vector_types.h>
#include <helper_math.h>

#include "boid.cuh"

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms

#define THREADS_PER_BLOCK 1024
#define BOID_COUNT 10 * THREADS_PER_BLOCK
#define BOID_POS_SIZE (BOID_COUNT * 2)
#define IS_BLOCK_NOT_FILLED (BOID_COUNT % THREADS_PER_BLOCK)
#define BLOCKS_COUNT (BOID_COUNT / THREADS_PER_BLOCK + IS_BLOCK_NOT_FILLED)

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width  = 512;
const unsigned int window_height = 512;

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

// boids
BoidSoA d_boids;

// parameters
// to pause and play the simulation
bool isActive = true;
// default weights
float separationWeight = 5.0f;
float alignmentWeight = 5.0f;
float cohesionWeight = 10.0f;
// default animation speed
float animationSpeed = 0.01f;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

// timer to measure fps
StopWatchInterface *timer = NULL;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

int *pArgc = NULL;
char **pArgv = NULL;

#define MAX(a,b) ((a > b) ? a : b)

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv, char *ref_file);
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

// Cuda functionality
void runCuda(struct cudaGraphicsResource **vbo_resource);
void checkResultCuda(int argc, char **argv, const GLuint &vbo);

const char *sSDKsample = "fishes";

float randFloatInRange(float min, float max);
void randomizeBoids();
void createBoids();
void freeBoids();

void launch_kernel(BoidSoA boidsoa, float4 *pos, float time)
{
    steerBoid<<<BLOCKS_COUNT, THREADS_PER_BLOCK>>>(boidsoa, pos, time, BOID_COUNT, separationWeight,
        alignmentWeight, cohesionWeight);
}

int main(int argc, char **argv)
{
    char *ref_file = NULL;

    pArgc = &argc;
    pArgv = argv;

#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    printf("%s starting...\n", sSDKsample);

    printf("Boid count: %d, blocks count: %d, threads per block: %d\n",
        BOID_COUNT, BLOCKS_COUNT, THREADS_PER_BLOCK);

    printf("\n");

    runTest(argc, argv, ref_file);

    printf("%s completed, returned %s\n", sSDKsample, (g_TotalErrors == 0) ? "OK" : "ERROR!");
    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

void computeFPS()
{
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);

        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "Fishes: %3.1f fps, cohesion (1,2): %3.1f, separation (3,4): %3.1f, alignment (5,6): %3.1f",
        avgFPS, cohesionWeight, separationWeight, alignmentWeight);
    glutSetWindowTitle(fps);
}

// initialize OpenGL
bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Fishes");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);

    // initialize necessary OpenGL extensions
    if (! isGLVersionSupported(2,0))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    // background color (navy blue)
    glClearColor(0.0, 0.0, 0.1, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

    SDK_CHECK_ERROR_GL();

    return true;
}

// Run a simple test for CUDA
bool runTest(int argc, char **argv, char *ref_file)
{
    // Create the CUTIL timer
    sdkCreateTimer(&timer);
    createBoids();

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);

    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    if (false == initGL(&argc, argv))
    {
        return false;
    }

    // register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutCloseFunc(cleanup);

    // create VBO
    createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

    // run the cuda part
    runCuda(&cuda_vbo_resource);

    // start rendering mainloop
    glutMainLoop();

    return true;
}

// Run the Cuda part of the computation
void runCuda(struct cudaGraphicsResource **vbo_resource)
{
    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
                                                         *vbo_resource));
    //printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

    if(isActive)
    {
        launch_kernel(d_boids, dptr, animationSpeed);
    }

    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags)
{
    assert(vbo);

    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = BOID_POS_SIZE * sizeof(float4);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

    SDK_CHECK_ERROR_GL();
}

void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{
    // unregister this buffer object with CUDA
    checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

// display callback to render graphics
void display()
{
    sdkStartTimer(&timer);

    // run CUDA kernel to generate vertex positions
    runCuda(&cuda_vbo_resource);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    // color fishes
    glColor3f(1.0, 1.0, 1.0);
    // draw fishes (head & tail) as a line
    glDrawArrays(GL_LINES, 0,  BOID_POS_SIZE);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();

    sdkStopTimer(&timer);
    computeFPS();
}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    }
}

// clean up allocated resources and write average fps
void cleanup()
{
    printf("Average FPS: %3.1f\n\n", avgFPS);

    sdkDeleteTimer(&timer);

    if (vbo)
    {
        deleteVBO(&vbo, cuda_vbo_resource);
    }

    freeBoids();
}

// called when a key is pressed
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27) :
            glutDestroyWindow(glutGetWindow());
            return;
        case ' ':
            isActive = !isActive;
            return;
        // cohesionWeight
        case '1':
            cohesionWeight -= 0.1;
            return;
        case '2':
            cohesionWeight += 0.1;
            return;

        // separationWeight
        case '3':
            separationWeight -= 0.1;
            return;
        case '4':
            separationWeight += 0.1;
            return;

        // alignmentWeight
        case '5':
            alignmentWeight -= 0.1;
            return;
        case '6':
            alignmentWeight += 0.1;
            return;


        // animationSpeed
        case '7':
            animationSpeed -= 0.001;
            return;
        case '8':
            animationSpeed += 0.001;
            return;
    }
}

// mouse event handler
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

// called when mouse is moved while a button is down
// used to move the camera
void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4)
    {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

////////////////////////////////////////////////////////////////////////////////
//! Check if the result is correct or write data to file for external
//! regression testing
////////////////////////////////////////////////////////////////////////////////
void checkResultCuda(int argc, char **argv, const GLuint &vbo)
{
    if (!d_vbo_buffer)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));

        // map buffer object
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        float *data = (float *) glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);

        // unmap GL buffer object
        if (!glUnmapBuffer(GL_ARRAY_BUFFER))
        {
            fprintf(stderr, "Unmap buffer failed.\n");
            fflush(stderr);
        }

        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo,
                                                     cudaGraphicsMapFlagsWriteDiscard));

        SDK_CHECK_ERROR_GL();
    }
}


// allocates memory for boids on GPU and initializes them with random values
void createBoids()
{
    checkCudaErrors(cudaMalloc(&d_boids.headingsX, sizeof(float) * BOID_COUNT));
    checkCudaErrors(cudaMalloc(&d_boids.headingsY, sizeof(float) * BOID_COUNT));
    checkCudaErrors(cudaMalloc(&d_boids.headingsZ, sizeof(float) * BOID_COUNT));
    
    checkCudaErrors(cudaMalloc(&d_boids.positionsX, sizeof(float) * BOID_COUNT));
    checkCudaErrors(cudaMalloc(&d_boids.positionsY, sizeof(float) * BOID_COUNT));
    checkCudaErrors(cudaMalloc(&d_boids.positionsZ, sizeof(float) * BOID_COUNT));

    cudaMemset(d_boids.headingsX, 0, sizeof(float) * BOID_COUNT);
    cudaMemset(d_boids.headingsY, 0, sizeof(float) * BOID_COUNT);
    cudaMemset(d_boids.headingsZ, 0, sizeof(float) * BOID_COUNT);

    cudaMemset(d_boids.positionsX, 0, sizeof(float) * BOID_COUNT);
    cudaMemset(d_boids.positionsY, 0, sizeof(float) * BOID_COUNT);
    cudaMemset(d_boids.positionsZ, 0, sizeof(float) * BOID_COUNT);

    checkCudaErrors(cudaMalloc(&d_boids.velocitiesX, sizeof(float) * BOID_COUNT));
    checkCudaErrors(cudaMalloc(&d_boids.velocitiesY, sizeof(float) * BOID_COUNT));
    checkCudaErrors(cudaMalloc(&d_boids.velocitiesZ, sizeof(float) * BOID_COUNT));

    randomizeBoids();
}

// generates random values for boids on CPU and copies them to GPU
void randomizeBoids()
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

    checkCudaErrors(cudaMemcpy(d_boids.positionsX, h_boids.positionsX, sizeof(float) * BOID_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_boids.positionsY, h_boids.positionsY, sizeof(float) * BOID_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_boids.positionsZ, h_boids.positionsZ, sizeof(float) * BOID_COUNT, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_boids.velocitiesX, h_boids.velocitiesX, sizeof(float) * BOID_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_boids.velocitiesY, h_boids.velocitiesY, sizeof(float) * BOID_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_boids.velocitiesZ, h_boids.velocitiesZ, sizeof(float) * BOID_COUNT, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_boids.headingsX, h_boids.headingsX, sizeof(float) * BOID_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_boids.headingsY, h_boids.headingsY, sizeof(float) * BOID_COUNT, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_boids.headingsZ, h_boids.headingsZ, sizeof(float) * BOID_COUNT, cudaMemcpyHostToDevice));

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

// generates random value in range [min, max)
float randFloatInRange(float min, float max)
{
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = max - min;
    float r = random * diff;
    return min + r;
}

// frees GPU memory
void freeBoids()
{
    cudaFree(d_boids.headingsX);
    cudaFree(d_boids.headingsY);
    cudaFree(d_boids.headingsZ);

    cudaFree(d_boids.positionsX);
    cudaFree(d_boids.positionsY);
    cudaFree(d_boids.positionsZ);

    cudaFree(d_boids.velocitiesX);
    cudaFree(d_boids.velocitiesY);
    cudaFree(d_boids.velocitiesZ);
}
