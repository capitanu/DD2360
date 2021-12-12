#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))



#define EPSILON 0.01

#define TPB 32
#define WIDTH 1600
#define HEIGHT 1600

double POSITION[] = {0.0, 0.0, 1.0};
#define RADIUS 1.5
const double COLOR[] = {0.0, 0.0, 1.0};
#define DIFFUSE 1
#define SPECULAR_C 1.0
#define SPECULAR_K 50.0

double L[] = {5.0, 5.0, -10.0};
const double COLOR_LIGHT[] = {1.0, 1.0, 1.0};
#define AMBIENT 0.05

double O[] = {0.0, 0.0, -1.0};
double Q[] = {0.0, 0.0, 0.0};

__device__ double POSITION_device[] = {0.0, 0.0, 1.0};
__device__ double O_device[] = {0.0, 0.0, -1.0};
__device__ double Q_device[] = {0.0, 0.0, 0.0};
__device__ double L_device[] = {5.0, 5.0, -10.0};
__device__ const double COLOR_LIGHT_device[] = {1.0, 1.0, 1.0};
__device__ const double COLOR_device[] = {0.0, 0.0, 1.0};
__device__ const double infinity = 0x7ff0000000000000;

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int round4(int x) {
    return x % 4 == 0 ? x : x - x % 4 + 4;
}

void writeBMP(int width, int height, char *image, int device)
{
	
    FILE *file;
	if(device == 1)
		file = fopen("cpu.bmp", "w+");
	else
		file = fopen("gpu.bmp", "w+");

	int bmp_size = 3 * height * width;

	int padded_width = round4(width * 3);

	int bitmap_size = height * padded_width * 3;
    char *bitmap = (char *) malloc(bitmap_size * sizeof(char));
	for(int row = 0; row < height; row++){
		for(int col = 0; col < width; col++){
			for(int color = 0; color < 3; color++){
				int index = row * padded_width + col * 3 + color;

				bitmap[index] = image[3*(row * width + col) + (2 - color)];
			}
		}
	}
    
	char tag[] = { 'B', 'M' };
	int header[] = {
		0,
		0, 0x36, 0x28,
		width, height,
		0x180001, 0, 0, 0x002e23, 0x002e23, 0, 0,
	};

	header[0] = sizeof(tag) + sizeof(header) + bmp_size;
	
	fwrite(&tag, sizeof(tag), 1, file);
	fwrite(&header, sizeof(header), 1, file);
	fwrite(image, sizeof(char) * bmp_size, 1, file);
    
    fclose(file);
}


void normalize(double *vector){
	double length = sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
	vector[0] /= length;
	vector[1] /= length;
	vector[2] /= length;
}

void normalize_minus(double* rtn,double *X, double *Y){
	rtn[0] = X[0] - Y[0];
	rtn[1] = X[1] - Y[1];
	rtn[2] = X[2] - Y[2];
	normalize(rtn);
}

void normalize_plus(double* rtn,double *X, double *Y){
	rtn[0] = X[0] + Y[0];
	rtn[1] = X[1] + Y[1];
	rtn[2] = X[2] + Y[2];
	normalize(rtn);
}

double dot(double *X, double *Y){
	return X[0] * Y[0] + X[1] * Y[1] + X[2] * Y[2];
}

double intersect_sphere(double *O, double *D){
	double a = dot(D, D);

	double OS[] = {O[0] - POSITION[0], O[1] - POSITION[1], O[2] - POSITION[2]};
	double b = 2 * dot(D, OS);
		
	double c = dot(OS, OS) - RADIUS * RADIUS;

	double disc = b * b - 4 * a * c;

	if(disc > 0){
		double discSqrt = sqrt(disc);
		double q = (b < 0) ? (-b - discSqrt) / 2.0 : (-b + discSqrt) / 2.0;
		double t0 = q / a;
		double t1 = c / q;
		double mint = min(t0, t1);
		double maxt = max(t0, t1);
		if(maxt >= 0){
			return (mint < 0) ? maxt : t0;
		}
	}
	return std::numeric_limits<double>::infinity();
}

double* trace_ray(double *O, double *D){
	double *col = (double *) malloc(3 * sizeof(double));

	double t = intersect_sphere(O,D);

	if(t == std::numeric_limits<double>::infinity()){
		return NULL;		
	}

	double M[] = {O[0] + D[0] * t, O[1] + D[1] * t, O[2] + D[2] * t};
	double *N = (double *)malloc( 3 * sizeof(double));
	normalize_minus(N, M, POSITION);
	double *toL = (double *)malloc( 3 * sizeof(double));
	normalize_minus(toL, L, M);
	double *toO = (double *)malloc( 3 * sizeof(double));
	normalize_minus(toO, O, M);

	double *toLtoO = (double *) malloc( 3 * sizeof(double));
	normalize_plus(toLtoO, toL, toO);

	col[0] = AMBIENT + DIFFUSE * max(dot(N, toL), 0.0) * COLOR[0] + SPECULAR_C * COLOR_LIGHT[0] * pow(max(dot(N, toLtoO), 0.0), SPECULAR_K);

	col[1] = AMBIENT + DIFFUSE * max(dot(N, toL), 0.0) * COLOR[1] + SPECULAR_C * COLOR_LIGHT[1] * pow(max(dot(N, toLtoO), 0.0), SPECULAR_K);

	col[2] = AMBIENT + DIFFUSE * max(dot(N, toL), 0.0) * COLOR[2] + SPECULAR_C * COLOR_LIGHT[2] * pow(max(dot(N, toLtoO), 0.0), SPECULAR_K);

	return col;
}

char* run_cpu(char *image){

	double x = (double) -1;
	double y = (double) -1;
	for(int i = 0; i < WIDTH; i++){
		for(int j = 0; j < HEIGHT; j++){
			Q[0] = x;
			Q[1] = y;
			y += (double) 2 / (double) HEIGHT;


			double D[] = {Q[0] - O[0], Q[1] - O[1], Q[2] - O[2]};

			normalize(D);

			
			double *col = trace_ray(O, D);

			if(col == NULL){
				continue;
			}

			image[(i + (j * HEIGHT)) * 3 + 2] = max(0.0, min(col[0], 1.0)) * 255;
			image[(i + (j * HEIGHT)) * 3 + 1] = max(0.0, min(col[1], 1.0)) * 255;
			image[(i + (j * HEIGHT)) * 3] = max(0.0, min(col[2], 1.0)) * 255;
			
		}
		x += (double) 2 / (double) WIDTH;
		y = (double) -1;
	}
	return image;
	
}

__device__ void gpu_normalize(double *vector){
	double length = sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
	vector[0] /= length;
	vector[1] /= length;
	vector[2] /= length;
}


__device__ void gpu_normalize_minus(double* rtn,double *X, double *Y){
	rtn[0] = X[0] - Y[0];
	rtn[1] = X[1] - Y[1];
	rtn[2] = X[2] - Y[2];
	gpu_normalize(rtn);
}

__device__ void gpu_normalize_plus(double* rtn,double *X, double *Y){
	rtn[0] = X[0] + Y[0];
	rtn[1] = X[1] + Y[1];
	rtn[2] = X[2] + Y[2];
	gpu_normalize(rtn);
}

__device__ double gpu_dot(double *X, double *Y){
	return X[0] * Y[0] + X[1] * Y[1] + X[2] * Y[2];
}

__device__ double gpu_intersect_sphere(double *D){
	double a = gpu_dot(D, D);
	double OS[] = {O_device[0] - POSITION_device[0], O_device[1] - POSITION_device[1], O_device[2] - POSITION_device[2]};
	
	double b = 2 * gpu_dot(D, OS);

	double c = gpu_dot(OS, OS) - RADIUS * RADIUS;

	double disc = b * b - 4 * a * c;
	if(disc > 0){
		double discSqrt = sqrt(disc);
		double q = (b < 0) ? (-b - discSqrt) / 2.0 : (-b + discSqrt) / 2.0;
		double t0 = q / a;
		double t1 = c / q;
		double mint = min(t0, t1);
		double maxt = max(t0, t1);
		if(maxt >= 0){
			return (mint < 0) ? maxt : t0;
		}
	}

	return infinity;
}

__device__ void gpu_trace_ray(double *col, double *D){

	double t = gpu_intersect_sphere(D);

	 if(t == infinity){
	 	col[0] = -1;
	 	return;
	 }

	double M_device[] = {O_device[0] + D[0] * t, O_device[1] + D[1] * t, O_device[2] + D[2] * t};
	double N_device[3], toO_device[3], toL_device[3], toLtoO_device[3];
	gpu_normalize_minus(N_device, M_device, POSITION_device);
	
	gpu_normalize_minus(toL_device, L_device, M_device);
	
	gpu_normalize_minus(toO_device, O_device, M_device);

	gpu_normalize_plus(toLtoO_device, toL_device, toO_device);

	col[0] = AMBIENT + DIFFUSE * max(gpu_dot(N_device, toL_device), 0.0) * COLOR_device[0] + SPECULAR_C * COLOR_LIGHT_device[0] * pow(max(gpu_dot(N_device, toLtoO_device), 0.0), SPECULAR_K);

	col[1] = AMBIENT + DIFFUSE * max(gpu_dot(N_device, toL_device), 0.0) * COLOR_device[1] + SPECULAR_C * COLOR_LIGHT_device[1] * pow(max(gpu_dot(N_device, toLtoO_device), 0.0), SPECULAR_K);

	col[2] = AMBIENT + DIFFUSE * max(gpu_dot(N_device, toL_device), 0.0) * COLOR_device[2] + SPECULAR_C * COLOR_LIGHT_device[2] * pow(max(gpu_dot(N_device, toLtoO_device), 0.0), SPECULAR_K);

}


__global__ void run_gpu(char *image){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(index > HEIGHT*WIDTH)
		return;
	
	int i = index % WIDTH;
	int j = index / WIDTH;

	double x = -1.0 + (double) i * 2.0/(double) WIDTH;
	double y = -1.0 + (double) j * 2.0/(double) HEIGHT;

	double D[] = {x - O_device[0], y - O_device[1], Q_device[2] - O_device[2]};

	gpu_normalize(D);
	
	double col[3];
	
	gpu_trace_ray(col, D);

	if(col[0] == -1){
		return;
	}

	image[index * 3 + 2] = max(0.0, min(col[0], 1.0)) * 255;
	image[index * 3 + 1] = max(0.0, min(col[1], 1.0)) * 255;
	image[index * 3] = max(0.0, min(col[2], 1.0)) * 255;

	
}

int main(int argc, char **argv)
{
	// CPU RayTracing
	char *cpu_image = (char *) malloc(sizeof(char) * HEIGHT * WIDTH * 3);

	double start_cpu = cpuSecond();
	run_cpu(cpu_image);
	printf("CPU time: %f\n", cpuSecond() - start_cpu);

	writeBMP(WIDTH, HEIGHT, cpu_image, 1);

	/* --------------------------------------------------*/

	// GPU RayTracing

	int ARRAY_SIZE = HEIGHT * WIDTH;
	int BLOCKS = (ARRAY_SIZE + TPB - 1) / TPB;

	char *d_image, *gpu_image;
	cudaMalloc(&d_image, WIDTH * HEIGHT * 3 * sizeof(char));
	gpu_image = (char*) malloc(3 * WIDTH * HEIGHT * sizeof(char));
	
	double start_gpu = cpuSecond();
	run_gpu<<<BLOCKS, TPB>>>(d_image);
	cudaDeviceSynchronize();
	printf("GPU time: %f\n", cpuSecond() - start_gpu);

	HANDLE_ERROR(cudaMemcpy(gpu_image, d_image, sizeof(char) * 3 * WIDTH * HEIGHT, cudaMemcpyDeviceToHost));
	
	writeBMP(WIDTH, HEIGHT, gpu_image, 2);

	int val = 0;
	for(int i = 0; i < 3 * WIDTH* HEIGHT; i++){
		if(abs(gpu_image[i] - cpu_image[i]) > EPSILON){
			val = 1;
		}
	}
	if(val == 1){
		printf("There are differences in the output images.\n");
	}
	else {
		printf("The results are correct.\n");
	}
	cudaFree(d_image);
	free(gpu_image);
	free(cpu_image);
    
    return 0;
}

