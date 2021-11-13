#include <stdio.h>

#define TPB 256


__global__ void helloWorldKernel(){
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Hello World! My threadId: %d\n", idx);
}

int main(){

	helloWorldKernel<<<1, TPB>>>();
	cudaDeviceSynchronize();
	return 0;
}
