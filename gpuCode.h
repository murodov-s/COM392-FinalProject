#ifndef GPULib
#define GPULib

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_texture_types.h>

#include "params.h"

#define cH2D            cudaMemcpyHostToDevice
#define cD2D            cudaMemcpyDeviceToDevice
#define cD2H            cudaMemcpyDeviceToHost

// use the same GPU_Palette for runmode 1 and runmode 2

struct myColors{
  float pink;
  float orange;
  float yellow;
};

struct GPU_Palette{

    unsigned int palette_width;
    unsigned int palette_height;
    unsigned long num_pixels; // population size
    unsigned long memSize;
    unsigned long memIntSize;
//    unsigned long memLongSize;

    myColors theColors;

    dim3 gThreads;
    dim3 gBlocks;

    curandState* rand;
    float* red;
    float* green;
    float* blue;




    // new stuff for modeling virus spread
    int* ming;   // how many other people a person mingles with per day
    float* susc; // susceptibility of a person (age-health)
    int* stage;  // stage of virus infection of a person (0-4)
};

// --------  virus simulation stuff
GPU_Palette initPopulation(void);
int updatePopulation(GPU_Palette* P, AParams* PARAMS, int day);

// kernel calls:
__global__ void whoDies(int* stage, curandState* gRand, float* susc, int sizePopulation);
//__global__ void recover(int* stage, curandState* gRand, int sizePopulation);
__global__ void recoverStage(int* stage, int infectPeriod, int sizePopulation);
__global__ void setMap(float* map, float val, int sizePopulation);
__global__ void mingle(int* stage, int* ming, float spreadrate, int duration,
            curandState* gRand, int sizePopulation);


__global__ void drawStage(float* red, float* green, float* blue,
  int* stage, int sizePopulation);




// -------- runmode 1 stuff for playing with pixel colors
GPU_Palette initGPUPalette(unsigned int height, unsigned int width);
int probeGPU(void);
int updatePalette(GPU_Palette*);
int freeGPUPalette(GPU_Palette* P1);

// kernel calls:
__global__ void updateReds(float* red, curandState* gRand);
__global__ void updateGreens(float* green);
__global__ void updateBlues(float* blue);
__global__ void initRands(curandState* state, unsigned long seed, unsigned long);

#endif
