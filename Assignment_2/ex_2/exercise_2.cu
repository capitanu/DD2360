#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define TPB 256

#define CPU true
#define GPU true
#define ARRAY_SIZE 1000000000

#define EPSILON 0.001

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void SAXPY_GPU(float *d_x, float *d_y, const float a){
	const int idx = threadIdx.x + blockDim.x * blockIdx.x;
	d_y[idx] = d_x[idx] * a + d_y[idx];
}

void SAXPY_CPU(float *x, float *y, const float a){
	for(int i = 0; i < ARRAY_SIZE; i++){
		y[i] = a*x[i] + y[i];
	}
}

int main(){

	float *x, *y, *y_gpu;
	x = (float *) malloc(sizeof(float) * ARRAY_SIZE);
	y = (float *) malloc(sizeof(float) * ARRAY_SIZE);
	y_gpu = (float *) malloc(sizeof(float) * ARRAY_SIZE);

	for(int i = 0; i < ARRAY_SIZE; i++){
		x[i] = rand() % 100;
		y[i] = rand() % 100;
	}
	float a = 3.45;
	
	float *d_x, *d_y;
	cudaMalloc(&d_x, sizeof(float) * ARRAY_SIZE);
	cudaMalloc(&d_y, sizeof(float) * ARRAY_SIZE);
	
	cudaMemcpy(d_x, x, sizeof(float) * ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, sizeof(float) * ARRAY_SIZE, cudaMemcpyHostToDevice);

	printf("Computing SAXPY on the CPU...");
	double start_cpu = cpuSecond();
	SAXPY_CPU(x,y,a);
	printf("Done! Took: %f seconds\n", cpuSecond() - start_cpu);

	printf("Computing SAXPY on the GPU...");
	double start_gpu = cpuSecond();
	SAXPY_GPU<<<(ARRAY_SIZE + TPB - 1)/TPB, TPB>>>(d_x, d_y, a);
	cudaDeviceSynchronize();
	printf("Done! Took: %f seconds\n", cpuSecond() - start_cpu);
	
	cudaMemcpy(y_gpu, d_y, sizeof(float) * ARRAY_SIZE, cudaMemcpyDeviceToHost);

	bool comp = true;
	for(int i = 0; i < ARRAY_SIZE; i++){
		if(abs(y[i] - y_gpu[i]) > EPSILON){
			comp = false;
			printf("%f\n", abs(y[i] - y_gpu[i]));
		}
	}
	
	printf("Comparing the output for each implementation...");
	if(comp)
		printf("Correct\n");
	else
		printf("Incorrect\n");
	
	

	return 0;
}
