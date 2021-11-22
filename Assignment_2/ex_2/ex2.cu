#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <cuda_profiler_api.h>
#include <stdlib.h>
//#define NUM_PARTICLES 10000
//#define NUM_ITERATIONS 100
//#define BLOCK_SIZE 1024

#define DT 1   //dt

unsigned long NUM_PARTICLES;
int NUM_ITERATIONS,BLOCK_SIZE;

typedef struct {
    float3 position;
    float3 velocity;
}Particle;

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}


__global__ void gpu(Particle* ref,float3* ref_acc,unsigned long *particle, int* iteration){
    /*
    * kernel
    */
    uint  NUM_PARTICLES=*particle;
    int  NUM_ITERATIONS= *iteration;

    const int idx = blockDim.x * blockIdx.x +threadIdx.x;
    if( idx >=NUM_PARTICLES) return ;
    for(int t=1; t<=NUM_ITERATIONS; t+=DT){
        //update position for this dt
        ref[idx].position.x+=ref[idx].velocity.x * DT;
        ref[idx].position.y+=ref[idx].velocity.y * DT;
        ref[idx].position.z+=ref[idx].velocity.z * DT;

        //update velocity for next dt
        ref[idx].velocity.x+=ref_acc->x;
        ref[idx].velocity.y+=ref_acc->y;
        ref[idx].velocity.z+=ref_acc->z;

    }
}

void initialization(Particle* x,float3* p_acc){
    /*initialize the particles*/
    for(int i=0; i <NUM_PARTICLES;++i){
        x[i].position.x=0;
        x[i].position.y=0;
        x[i].position.z=0;

        x[i].velocity.x=0;
        x[i].velocity.y=0;
        x[i].velocity.z=0;

    }

    /*initialize the accelerator*/
    p_acc->x=1.0f;
    p_acc->y=2.0f;
    p_acc->z=3.0f;

}

double test_gpu(Particle* x,float3 acc,Particle* res){
    /*
    lanuch the kernel
    */
   
   double iStart = cpuSecond();
    Particle* ref=NULL;
    float3 *ref_acc=NULL;
    cudaProfilerStart();
    cudaMalloc(&ref,NUM_PARTICLES*sizeof(Particle));
    cudaMalloc(&ref_acc,sizeof(float3));

    unsigned long* particle;
    int  *iteration;
    cudaMalloc(&particle,sizeof(unsigned long));
    cudaMalloc(&iteration,sizeof(int));
    
    

    cudaMemcpy(ref,x,NUM_PARTICLES*sizeof(Particle),cudaMemcpyHostToDevice);
    cudaMemcpy(ref_acc,&acc,sizeof(float3),cudaMemcpyHostToDevice);

    cudaMemcpy(particle,&NUM_PARTICLES,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(iteration,&NUM_ITERATIONS,sizeof(int),cudaMemcpyHostToDevice);
    
    int grid_size=(int)((NUM_PARTICLES+BLOCK_SIZE-1)/BLOCK_SIZE);

    
    
    
    dim3 grid(grid_size);
    dim3 block(BLOCK_SIZE);


    


    gpu<<< grid,block>>>(ref,ref_acc,particle,iteration);
    cudaDeviceSynchronize();
    
    

    
    
    
    cudaMemcpy(res,ref,NUM_PARTICLES*sizeof(Particle),cudaMemcpyDeviceToHost);
    double iElaps = cpuSecond() - iStart;

    cudaFree(ref);
    cudaFree(ref_acc);
    cudaFree(particle);
    cudaFree(iteration);

    
    cudaProfilerStop();
    

    
    return iElaps; 

}

double test_cpu(Particle* x,float3 acc, Particle* res){
    for(int i=0; i <NUM_PARTICLES; ++i){
        res[i]=x[i];
    }

    double iStart = cpuSecond();
    for(int t=1; t<=NUM_ITERATIONS; t+=DT){
        for(int i=0; i<NUM_PARTICLES; ++i){
            res[i].position.x+=res[i].velocity.x * DT;
            res[i].position.y+=res[i].velocity.y * DT;
            res[i].position.z+=res[i].velocity.z * DT;

            //update velocity for next dt
            res[i].velocity.x+=acc.x;
            res[i].velocity.y+=acc.y;
            res[i].velocity.z+=acc.z;
        }
    }
    double iElaps = cpuSecond() - iStart;
    return iElaps;

}

int main(int argc, char** argv){
    
    
    printf("particle number: %s\n",argv[1]);
    printf("iteration number: %s\n",argv[2]);
    printf("block size: %s\n",argv[3]);

    NUM_PARTICLES =  (unsigned long)atoi(argv[1]);
    printf("%ld\n",NUM_PARTICLES);
    NUM_ITERATIONS = atoi(argv[2]);
    BLOCK_SIZE =  atoi(argv[3]);
    
    Particle x[NUM_PARTICLES],x_gpu[NUM_PARTICLES],x_cpu[NUM_PARTICLES];
    float3 acc;//acc changes the velocity
    initialization(x,&acc);
    double elapse_cpu = test_cpu(x,acc,x_cpu);
    double elapse_gpu = test_gpu(x,acc,x_gpu);

    for(int i=0; i < NUM_PARTICLES ;++i){
        assert(x_gpu[i].position.x==x_cpu[i].position.x);
        assert(x_gpu[i].position.y==x_cpu[i].position.y);
        assert(x_gpu[i].position.z==x_cpu[i].position.z);
        assert(x_gpu[i].velocity.x==x_cpu[i].velocity.x);
        assert(x_gpu[i].velocity.y==x_cpu[i].velocity.y);
        assert(x_gpu[i].velocity.z==x_cpu[i].velocity.z);

    } 




    
    
    printf("gpu execution time is %f\n",elapse_gpu);
    printf("cpu execution time is %f\n",elapse_cpu);
    
    //cudaProfilerStop();
    return 0;
}