#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cstdlib>
#include <curand_kernel.h>
#include <curand.h>

//#define TPB         256
//#define NUM_ITER    10
#define SEED        921
#define NUM_THREADS 10000000

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void computeProbabilityKernel(curandState *states, int *d_count, int i){
	const int idx = threadIdx.x + blockDim.x * blockIdx.x;

	curand_init((SEED + i) * idx, idx, 0, &states[idx]);

	float x = curand_uniform(&states[idx]);
	float y = curand_uniform(&states[idx]);

	float z = sqrt((x*x) + (y*y));

	if(z <= 1.0){
		d_count[idx] = 1;
	}	
	
}

int main(int argc, char* argv[]){

	const int TPB = atoi(argv[2]);
	const int BLOCKS = (NUM_THREADS + TPB - 1)/TPB;
	const int NUM_ITER = atoi(argv[1]);

	curandState *d_random;
	cudaMalloc((void**)&d_random, TPB*BLOCKS*sizeof(curandState));

	int *count = (int*) calloc(BLOCKS*TPB, sizeof(int));
	int *d_count;
	cudaMalloc(&d_count, sizeof(int) * BLOCKS * TPB);


	float count_1s = 0;
	double start_gpu = cpuSecond();
	
	for(int i = 0; i < NUM_ITER; i++){

		for(int j = 0; j < TPB*BLOCKS; j++){
			count[j] = 0;
		}
		cudaMemcpy(d_count, count, sizeof(int) * BLOCKS * TPB, cudaMemcpyHostToDevice);

		computeProbabilityKernel<<<BLOCKS,TPB>>>(d_random, d_count, i);
		cudaDeviceSynchronize();
		
		cudaMemcpy(count, d_count, sizeof(int) * TPB * BLOCKS, cudaMemcpyDeviceToHost);

		for(int j = 0; j < BLOCKS*TPB; j++){
			if(count[j] == 1){
				count_1s += 1.0;
			}
		}

		float pi =  count_1s * 4.0 / ((float)(i+1) * (float) NUM_THREADS);
		printf("%f ", pi);
	}
	//printf("Time: %f seconds with %d threads and %d iterations per threads\n", cpuSecond() - start_gpu, NUM_THREADS, NUM_ITER);
	//printf("%f",cpuSecond()-start_gpu);
	//float pi =  count_1s * 4.0 / ((float) NUM_ITER * (float) NUM_THREADS);
	
    //printf("The result is %f\n", pi);
	
	return 0;
}