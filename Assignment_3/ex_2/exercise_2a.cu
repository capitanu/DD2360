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

	const int BLOCK_SIZE = (NUM_PARTICLES + TPB - 1) / TPB;

	Particle *particles = (Particle *) calloc(NUM_PARTICLES, sizeof(Particle));
	Particle *d_particles;

	double3 rands;
	
	//cudaMallocHost(&d_particles, sizeof(Particle) * NUM_PARTICLES);
	cudaMalloc(&d_particles, sizeof(Particle) * NUM_PARTICLES);


	double start_gpu = cpuSecond();
	for(int i = 0; i < NUM_ITERATIONS; i++){
		cudaMemcpy(d_particles, particles, sizeof(Particle) * NUM_PARTICLES, cudaMemcpyHostToDevice);

		rands.x = (double) rand() / (double) RAND_MAX;
		rands.y = (double) rand() / (double) RAND_MAX;
		rands.z = (double) rand() / (double) RAND_MAX;

		timestepKernel<<<BLOCK_SIZE, TPB>>>(d_particles, rands);
		
		cudaMemcpy(particles, d_particles, sizeof(Particle) * NUM_PARTICLES, cudaMemcpyDeviceToHost);
		
		cudaDeviceSynchronize();
	}
	printf("%f ", cpuSecond() - start_gpu);
	
	cudaFree(d_particles);
	
	double start_cpu = cpuSecond();
	for(int i = 0; i < NUM_ITERATIONS; i++){
		rands.x = (double) rand() / (double) RAND_MAX;
		rands.y = (double) rand() / (double) RAND_MAX;
		rands.z = (double) rand() / (double) RAND_MAX;
		timestepCPU(particles, rands, NUM_PARTICLES);
	}
	printf("%f\n", cpuSecond() - start_cpu);

	free(particles);

	
	return 0;
}