/*******************************************************************************
*
*   COMMENTS GO HERE
*
*   TODO LIST GOES HERE
*
*******************************************************************************/
#include <cuda.h>
#include <stdio.h>
#include "gpuCode.h"
#include "params.h"

texture<float, 2> texBlue;
texture<int, 2> texStage;

/******************************************************************************/
// VIRUS SIMULATION CODE
/******************************************************************************/
GPU_Palette initPopulation(void) // for simulating virus
{
  GPU_Palette X;

  X.gThreads.x = 32;  // 32 x 32 = 1024 threads per block
  X.gThreads.y = 32;
  X.gThreads.z = 1;
  X.gBlocks.x = 32;  // 32 x 32 = 1024 blocks
  X.gBlocks.y = 32;
  X.gBlocks.z = 1;

  X.palette_width = 1024;       // save this info
  X.palette_height = 1024;
  X.num_pixels = 1024*1024; // 1048576
  X.memSize =  1024*1024 * sizeof(float);
  X.memIntSize =  1024*1024 * sizeof(int);

  // keep color stuff for visualizing virus spread
  cudaError_t err;
  err = cudaMalloc((void**) &X.red, X.memSize);
  if(err != cudaSuccess){
    printf("cuda error allocating red = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }
  err = cudaMalloc((void**) &X.green, X.memSize);
  if(err != cudaSuccess){
    printf("cuda error allocating green = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }
  err = cudaMalloc((void**) &X.blue, X.memSize);  // b
  if(err != cudaSuccess){
    printf("cuda error allocating blue = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }
  err = cudaMalloc((void**) &X.rand, X.num_pixels * sizeof(curandState));
  if(err != cudaSuccess){
    printf("cuda error allocating blue = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }

  // for initializing population with (random) susceptibility ratings
  err = cudaMalloc((void**) &X.susc, X.memSize);
  if(err != cudaSuccess){
    printf("cuda error allocating susc = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }
  err = cudaMalloc((void**) &X.stage, X.memIntSize);
  if(err != cudaSuccess){
    printf("cuda error allocating stage = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }
  err = cudaMalloc((void**) &X.ming, X.memIntSize);
  if(err != cudaSuccess){
    printf("cuda error allocating ming = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }


  initRands <<< X.gBlocks, X.gThreads >>> (X.rand, time(NULL), X.num_pixels);

  cudaChannelFormatDesc desc= cudaCreateChannelDesc <float>();
  unsigned int pitch = sizeof(float)*1024;
  cudaBindTexture2D(NULL, texBlue, X.blue, desc, 1024, 1024, pitch);

  cudaChannelFormatDesc desc2= cudaCreateChannelDesc <int>();
  unsigned int pitch2 = sizeof(int)*1024;
  cudaBindTexture2D(NULL, texStage, X.stage, desc2, 1024, 1024, pitch2);

  return X;
}

/******************************************************************************/
// analogous to updatePalette in runmode 1
int updatePopulation(GPU_Palette* P, AParams* PARAMS, int day){

  mingle <<< P->gBlocks, P->gThreads >>> (P->stage, P->ming, PARAMS->spreadrate,
                      PARAMS->duration, P->rand, P->num_pixels);

  recoverStage <<< P->gBlocks, P->gThreads >>> (P->stage, PARAMS->duration, P->num_pixels);

  whoDies <<< P->gBlocks, P->gThreads >>> (P->stage, P->rand, P->susc, P->num_pixels);

  drawStage <<< P->gBlocks, P->gThreads >>> (P->red, P->green, P->blue,
          P->stage, P->num_pixels);

  return 0;
}

/******************************************************************************/
__global__ void setMap(float* map, float val, long sizePopulation){

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int tid = x + (y * blockDim.x * gridDim.x);

  if (tid < sizePopulation){
    map[tid] = val;
  }
}


/******************************************************************************/
__global__ void recoverStage(int* stage, int infectPeriod, int sizePopulation){

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int tid = x + (y * blockDim.x * gridDim.x);

  if (tid < sizePopulation){
    if(stage[tid] > 0 && stage[tid] < 100){ // if in infectious stage
      if(stage[tid] >= infectPeriod) stage[tid] = 100; // if done with infectious stage, move to recovery stage (stage 100)
      else stage[tid] += 1; // increment day by 1
    }
  }
}

/******************************************************************************/
__global__ void whoDies(int* stage, curandState* gRand, float* susc, int sizePopulation){

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int tid = x + (y * blockDim.x * gridDim.x);

  float theRand;
  curandState localState;
  if (tid < sizePopulation){
    if(stage[tid] >= 100 && stage[tid] < 200){ // if in recovery
      stage[tid] += 1; // increment day
      if(stage[tid] > 107) stage[tid] = 200; // if made it past last day of recovery, immune
      if(stage[tid] == 107){  // if on last day of recovery, decide if live or doe

        // generate noise
        localState = gRand[tid];
        theRand = curand_uniform(&localState); // value between 0-1
        gRand[tid] = localState;

        if(theRand < susc[tid]) stage[tid] = -1;
        if(theRand < susc[tid]) stage[tid] = -1;
//        if(theRand < 1.0) stage[tid] = -1;
      }
    }
  }
}


/******************************************************************************/
__global__ void mingle(int* stage, int* ming, float spreadrate, int duration,
            curandState* gRand, int sizePopulation){

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int tid = x + (y * blockDim.x * gridDim.x);

  float theRand;
  int nX;
  int nY;
  int neighborStage;
  if (tid < sizePopulation){ // should never need this, but good practice
    if(stage[tid] != 0) return; // cannot be infected

    //meet a random number of people based on ming score
    for(long i = 0; i < ming[tid]; i++){ // meet this many people per day

      // generate noise
      curandState localState = gRand[tid];
      theRand = curand_uniform(&localState); // value between 0-1
      gRand[tid] = localState;
      nX = floor(theRand * 1024.0);

      localState = gRand[tid];
      theRand = curand_uniform(&localState);
      gRand[tid] = localState;
      nY = floor(theRand * 1024.0);

      neighborStage = tex2D(texStage, nX, nY);

      // if the random neighbor is infected, self gets infected with some probability
      localState = gRand[tid];
      theRand = curand_uniform(&localState);
      gRand[tid] = localState;
      if(neighborStage > 0 && neighborStage < duration){ // if neighbor is infectious
        if(theRand < spreadrate) stage[tid] = 1;  // self becomes infected with some probability
        }
    }
  }
}


/******************************************************************************/
__global__ void drawStage(float* red, float* green, float* blue,
  int* stage, int sizePopulation){

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int tid = x + (y * blockDim.x * gridDim.x);

  if (tid < sizePopulation){
    if (stage[tid] == 0) {  // if not infected, draw as white
      red[tid] = 1.0;
      green[tid] = 1.0;
      blue[tid] = 1.0;
      }
    else if (stage[tid] == -1) {  // if dead, draw in black
      red[tid] = 0.0;
      green[tid] = 0.0;
      blue[tid] = 0.0;
      }
    else if (stage[tid] < 100) { // if in infectious stage, draw as red
      red[tid] = 1.0;
      green[tid] = 0.0;
      blue[tid] = 0.0;
      }
    else if (stage[tid] < 200) { // if in recovery stage, draw as green
      red[tid] = 0.0;
      green[tid] = 1.0;
      blue[tid] = 0.0;
      }
    else if (stage[tid] < 300) { // if in immune, draw as blue
      red[tid] = 0.0;
      green[tid] = 0.0;
      blue[tid] = 1.0;
      }
    }
}




/******************************************************************************/
// RUNMODE 0 CODE
/******************************************************************************/
// return information about CUDA GPU devices on this machine
int probeGPU(){

  cudaError_t err;
  err = cudaDeviceReset();

  cudaDeviceProp prop;
  int count;
  err = cudaGetDeviceCount(&count);
  if(err != cudaSuccess){
    printf("problem getting device count = %s\n", cudaGetErrorString(err));
    return 1;
    }
  printf("number of GPU devices: %d\n\n", count);

  for (int i = 0; i< count; i++){
    printf("************ GPU Device: %d ************\n\n", i);
    err = cudaGetDeviceProperties(&prop, i);
    if(err != cudaSuccess){
      printf("problem getting device properties = %s\n", cudaGetErrorString(err));
      return 1;
      }

    printf("\tName: %s\n", prop.name);
    printf( "\tCompute capability: %d.%d\n", prop.major, prop.minor);
    printf( "\tClock rate: %d\n", prop.clockRate );
    printf( "\tDevice copy overlap: " );
      if (prop.deviceOverlap)
        printf( "Enabled\n" );
      else
        printf( "Disabled\n" );
    printf( "\tKernel execition timeout: " );
      if (prop.kernelExecTimeoutEnabled)
        printf( "Enabled\n" );
      else
        printf( "Disabled\n" );
    printf( "--- Memory Information for device %d ---\n", i );
    printf("\tTotal global mem: %ld\n", prop.totalGlobalMem );
    printf("\tTotal constant Mem: %ld\n", prop.totalConstMem );
    printf("\tMax mem pitch: %ld\n", prop.memPitch );
    printf( "\tTexture Alignment: %ld\n", prop.textureAlignment );
    printf("\n");
    printf( "\tMultiprocessor count: %d\n", prop.multiProcessorCount );
    printf( "\tShared mem per processor: %ld\n", prop.sharedMemPerBlock );
    printf( "\tRegisters per processor: %d\n", prop.regsPerBlock );
    printf( "\tThreads in warp: %d\n", prop.warpSize );
    printf( "\tMax threads per block: %d\n", prop.maxThreadsPerBlock );
    printf( "\tMax block dimensions: (%d, %d, %d)\n",
                  prop.maxThreadsDim[0],
                  prop.maxThreadsDim[1],
                  prop.maxThreadsDim[2]);
    printf( "\tMax grid dimensions: (%d, %d, %d)\n",
                  prop.maxGridSize[0],
                  prop.maxGridSize[1],
                  prop.maxGridSize[2]);
    printf("\n");
  }

return 0;
}






/******************************************************************************/
// RUNMODE 1 CODE
/******************************************************************************/
int updatePalette(GPU_Palette* P){

  updateReds <<< P->gBlocks, P->gThreads >>> (P->red, P->rand);
  updateGreens <<< P->gBlocks, P->gThreads >>> (P->green);
	updateBlues <<< P->gBlocks, P->gThreads >>> (P->blue);

  return 0;
}

/******************************************************************************/
//__global__ void updateReds(float* red){
__global__ void updateReds(float* red, curandState* gRand){

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int tid = x + (y * blockDim.x * gridDim.x);

  // generate noise
  curandState localState = gRand[tid];
  float theRand = curand_uniform(&localState); // value between 0-1
//  float theRand = curand_poisson(&localState, .5);
  gRand[tid] = localState;

  // sparkle the reds:
  if(theRand > .999) red[tid] = red[tid] *.9;
  else if(theRand < .001) red[tid] = (1.0-red[tid]);
}

/******************************************************************************/
__global__ void updateGreens(float* green){

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int tid = x + (y * blockDim.x * gridDim.x);

  green[tid] = green[tid] *.888;
//  green[tid] = green[tid] * 0;
}

/******************************************************************************/
__global__ void initRands(curandState* state, unsigned long seed, unsigned long numPixels){

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int tid = x + (y * blockDim.x * gridDim.x);

  if(tid < numPixels) curand_init(seed, tid, 0, &state[tid]);

}



/******************************************************************************/
__global__ void updateBlues(float* blue){

  int x = threadIdx.x + (blockIdx.x * blockDim.x);
  int y = threadIdx.y + (blockIdx.y * blockDim.y);
  int tid = x + (y * blockDim.x * gridDim.x);

  // find neighborhood average blue value
  float acc = 0.0;
  for (int i = -20; i <= 20; i++){      // 11 pixels-threads in x direction
    for (int j = -20; j <= 20; j++){    // 11 pixels-threads in the y direction
      acc += tex2D(texBlue, x+i, y+j);
    }
  }
  acc /= 241.0;

  blue[tid] = acc;

}


/******************************************************************************/
GPU_Palette initGPUPalette(unsigned int imageWidth, unsigned int imageHeight)
{
  GPU_Palette X;

  X.gThreads.x = 32;  // 32 x 32 = 1024 threads per block
  X.gThreads.y = 32;
  X.gThreads.z = 1;
  X.gBlocks.x = ceil(imageWidth/32);  // however many blocks needed for image
  X.gBlocks.y = ceil(imageHeight/32);
  X.gBlocks.z = 1;

  X.palette_width = imageWidth;       // save this info
  X.palette_height = imageHeight;
  X.num_pixels = imageWidth * imageHeight;
  X.memSize =  imageWidth * imageHeight * sizeof(float);

  // allocate memory on GPU
  cudaError_t err;
  err = cudaMalloc((void**) &X.red, X.memSize);
  if(err != cudaSuccess){
    printf("cuda error allocating red = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }
  err = cudaMalloc((void**) &X.green, X.memSize);
  if(err != cudaSuccess){
    printf("cuda error allocating green = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }
  err = cudaMalloc((void**) &X.blue, X.memSize);  // b
  if(err != cudaSuccess){
    printf("cuda error allocating blue = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }

  err = cudaMalloc((void**) &X.rand, X.num_pixels * sizeof(curandState));
  if(err != cudaSuccess){
    printf("cuda error allocating blue = %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }

  initRands <<< X.gBlocks, X.gThreads >>> (X.rand, time(NULL), X.num_pixels);

  cudaChannelFormatDesc desc= cudaCreateChannelDesc <float>();
  unsigned int pitch = sizeof(float)*imageWidth;
  cudaBindTexture2D(NULL, texBlue, X.blue, desc, imageWidth, imageHeight, pitch);


  return X;
}



/******************************************************************************/
int freeGPUPalette(GPU_Palette* P) {

  // free gpu memory
//  cudaFree(P->gray);
  cudaFree(P->red);
  cudaFree(P->green);
  cudaFree(P->blue);

  cudaUnbindTexture(texBlue);

  return 0;
}

/*************************************************************************/
