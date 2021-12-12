// Template file for the OpenCL Assignment 4

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define size 10000000
#define EPSILON 0.001


double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void SAXPY_CPU(float *x, float *y, const float a){
	for(int i = 0; i < size; i++){
		y[i] = a*x[i] + y[i];
	}
}


int main(int argc, char *argv) {	


	float *x, *y, *y_gpu;
	x = (float *) malloc(sizeof(float) * size);
	y = (float *) malloc(sizeof(float) * size);
	y_gpu = (float *) malloc(sizeof(float) * size);

	for(int i = 0; i < size; i++){
		x[i] = rand() % 100;
		y[i] = rand() % 100;
	}
	float a = 3.45;	

	double gpu_starttime = cpuSecond();

#pragma acc parallel loop copyin(y[0:size]) copyin(x[0:size]) copyout(y_gpu[0:size])
	for(int i = 0; i < size; i++){
		y_gpu[i] = a*x[i] + y[i];
	}
	
	printf("OpenACC time: %f\n", cpuSecond() - gpu_starttime);

	double cpu_starttime = cpuSecond();
	SAXPY_CPU(x, y, a);
	printf("CPU time: %f\n", cpuSecond() - cpu_starttime);


	int comp = 1;
	for(int i = 0; i < size; i++){
		printf("%f\n", y_gpu[i]);
		if(abs(y[i] - y_gpu[i]) > EPSILON){
			comp = 0;
			printf("%f\n", abs(y[i] - y_gpu[i]));
		}
	}

	printf("Comparing the output for each implementation...");
	if(comp == 1)
		printf("Correct\n");
	else
		printf("Incorrect\n");

	
	free(y_gpu);
	free(y);
	free(x);
  
	return 0;
}
