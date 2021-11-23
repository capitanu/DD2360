#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cstdlib>

//#define NUM_PARTICLES 10000
//#define NUM_ITERATIONS 200
//#define TPB 64

struct Particle{
	float3 position;
	float3 velocity;
};

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void timestepKernel(Particle *d_particles, double3 randoms){
	const int idx = threadIdx.x + blockDim.x * blockIdx.x;

	d_particles[idx].velocity.x = randoms.x;
	d_particles[idx].velocity.y = randoms.y;
	d_particles[idx].velocity.z = randoms.z;

	d_particles[idx].position.x = d_particles[idx].position.x + d_particles[idx].velocity.x;
	d_particles[idx].position.y = d_particles[idx].position.y + d_particles[idx].velocity.y;
	d_particles[idx].position.z = d_particles[idx].position.z + d_particles[idx].velocity.z;
	
}

void timestepCPU(Particle *particles, double3 randoms, const int num_particles){
	for(int idx = 0; idx < num_particles; idx++){
				
		particles[idx].velocity.x = randoms.x;
		particles[idx].velocity.y = randoms.y;
		particles[idx].velocity.z = randoms.z;

		particles[idx].position.x = particles[idx].position.x + particles[idx].velocity.x;
		particles[idx].position.y = particles[idx].position.y + particles[idx].velocity.y;
		particles[idx].position.z = particles[idx].position.z + particles[idx].velocity.z;
	
	}
}

int main(int argc, char* argv[]){

	const int NUM_PARTICLES = atoi(argv[1]);
	const int NUM_ITERATIONS = atoi(argv[2]);
	const int TPB = atoi(argv[3]);
	const int BATCH_SIZE = atoi(argv[4]);
	const int NUM_STREAMS = atoi(argv[5]);

	const int BLOCK_SIZE = (NUM_PARTICLES + TPB - 1) / TPB;

	Particle *particles = (Particle *) calloc(NUM_PARTICLES, sizeof(Particle));
	Particle *d_particles;

	cudaStream_t stream[4];
	cudaStreamCreate(&stream[0]);
	cudaStreamCreate(&stream[1]);
	cudaStreamCreate(&stream[2]);
	cudaStreamCreate(&stream[3]);

	

	double3 rands;
	
	cudaMalloc(&d_particles, sizeof(Particle) * NUM_PARTICLES);



		
	double start_gpu = cpuSecond();
	for(int i = 0; i < NUM_ITERATIONS; i++){

		for(int j = 0; j < NUM_STREAMS; j++){
			
			cudaMemcpyAsync(&d_particles[BATCH_SIZE * j], &particles[BATCH_SIZE * j], sizeof(Particle) * BATCH_SIZE, cudaMemcpyHostToDevice, stream[j]);
			rands.x = (double) rand() / (double) RAND_MAX;
			rands.y = (double) rand() / (double) RAND_MAX;
			rands.z = (double) rand() / (double) RAND_MAX;

			timestepKernel<<<BLOCK_SIZE, TPB, 0, stream[j]>>>(d_particles, rands);
		
		}
	}
	printf("%f\n", cpuSecond() - start_gpu);
	
	cudaFree(d_particles);	

	cudaStreamDestroy(stream[0]);
	cudaStreamDestroy(stream[1]);
	cudaStreamDestroy(stream[2]);
	cudaStreamDestroy(stream[3]);
	free(particles);

	
	return 0;
}