#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#define HISTOGRAM_LENGTH 256
#define YSIZE 600
#define XSIZE 1200
#define TrainDataNum 7736
#define DataLen 1024
#define DataWidth 32
#define KernelWidth 5
#define KernelRadius 2
#define TileWidth 8
#define BlockWidth TileWidth+KernelWidth-1
#define RoutingX 79
#define RoutingWidth 512
#define RoutingY 500
#define RoutingHeight 55
#define MoneyX 922
#define MoneyWidth 200
#define MoneyY 218
#define MoneyHeight 64
#define NumX 1010
#define NumWidth 128
#define NumY 53
#define NumHeight 64
#define Boundary 39
#define BLACK_THRESHOLD 80
#define VERIFY_BOUNDARY 15000
#define RED_THRESHOLD 176
#define GREEN_THRESHOLD 176
#define BLUE_THRESHOLD 127

void cuCheck(int line) {
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Error: %s, %d\n", cudaGetErrorString(err), line);
    }
}
//nvcc -arch sm_20 checkReader2.cu

//knn code below{{{{{{{{{{{{{{{{{{{{{{{{
int* digits=NULL;
float* trainDataHost=NULL;
float* trainDataKernel=NULL;
float* distantHost=NULL;
float* distantKernel=NULL;


__global__ void knn(float* trainDataKernel, float* distantKernel, int count) {
	__shared__ float digit[DataWidth][DataWidth];
	__shared__ float train[DataWidth][DataWidth];
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int bx=blockIdx.x;

	train[tx][ty]=trainDataKernel[bx*DataLen+ty*DataWidth+tx];

	for(int i=0; i<count; ++i) {
		digit[tx][ty]=testDigit[i*DataLen+ty*DataWidth+tx];
		__syncthreads();

		float cur=digit[tx][ty]-train[tx][ty];
		cur=cur*cur;
		digit[tx][ty]=cur;
		for(int stride=16; stride>0; stride/=2) {
			__syncthreads();
			if(tx<stride&&ty<stride) {
				digit[tx][ty]+=digit[tx+stride][ty]+digit[tx+stride][ty+stride]+digit[tx][ty+stride];
			}
		}
		__syncthreads();
		if(tx==0) {
			distantKernel[TrainDataNum*i+bx]=int(digit[0][0])+float(bx)/10000;
		}

		__syncthreads();
	}
}


void initKNN() {
	digits=(int*)malloc(TrainDataNum*sizeof(int));
	initDigits(digits);

	int trainDataSize=sizeof(float)*DataLen*TrainDataNum;
	trainDataHost=(float*)malloc(trainDataSize);
	initTrainDataHost(trainDataHost);
	cudaMalloc((void**) &trainDataKernel, trainDataSize);
	cuCheck(__LINE__);
	cudaMemcpy(trainDataKernel, trainDataHost, trainDataSize, cudaMemcpyHostToDevice);
	cuCheck(__LINE__);

	int distantSize=sizeof(float)*TrainDataNum*15;
	distantHost=(float*)malloc(distantSize);
	cudaMalloc((void**) &distantKernel, distantSize);
	cuCheck(__LINE__);
}


void freeKNN() {
	free(digits);
	cudaFree(trainDataKernel);
	free(trainDataHost);
	free(distantHost);
	cudaFree(distantKernel);
}


__global__ void matrixMultiply_device(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
	__shared__ float sharedA[16][16];
	__shared__ float sharedB[16][16];
	int Row=blockIdx.y*16+threadIdx.y;
	int Column=blockIdx.x*16+threadIdx.x;
	float resultC=0.0f;
	for (int i=0; i<ceil(numAColumns/16.0); i++)
	{
		if (Row<numARows && i*16+threadIdx.x<numAColumns)
			sharedA[threadIdx.y][threadIdx.x]=A[Row*numAColumns+i*16+threadIdx.x];
		else
			sharedA[threadIdx.y][threadIdx.x]=0.0f;
		if (i*16+threadIdx.y<numBRows && Column<numBColumns)
			sharedB[threadIdx.y][threadIdx.x]=B[(i*16+threadIdx.y)*numBColumns+Column];
		else
			sharedB[threadIdx.y][threadIdx.x]=0.0f;
		__syncthreads();
		for (int j=0; j<16; j++)
		{
			resultC+=sharedA[threadIdx.y][j]*sharedB[j][threadIdx.x];
		}
		__syncthreads();
	}
	if (Row<numCRows && Column<numCColumns)
		C[Row*numCColumns+Column]=resultC;
	return;
}

float* matrixMultiply_host(float* hostA, float* hostB) {
	float *hostC;
	float *deviceA;
	float *deviceB;
	float *deviceC;
	int numARows=1934;
	int numAColumns=1024;
	int numBRows=numAColumns;
	int numBColumns=64;
	int numCRows=numARows;
	int numCColumns=numBColumns;

	hostC=(float*)malloc(numCRows*numCColumns*sizeof(float));
	cudaMalloc((void**)&deviceA, numARows*numAColumns*sizeof(float));
	cudaMalloc((void**)&deviceB, numBRows*numBColumns*sizeof(float));
	cudaMalloc((void**)&deviceC, numCRows*numCColumns*sizeof(float));
	cuCheck(__LINE__);

	cudaMemcpy(deviceA, hostA, numARows*numAColumns*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, hostB, numBRows*numBColumns*sizeof(float), cudaMemcpyHostToDevice);
	cuCheck(__LINE__);

	dim3 DimGrid(ceil(numCColumns/16.0),ceil(numCRows/16.0),1);
	dim3 DimBlock(16,16,1);
	matrixMultiply_device<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
	cudaDeviceSynchronize();
	cuCheck(__LINE__);

	cudaMemcpy(hostC, deviceC, numCRows*numCColumns*sizeof(float), cudaMemcpyDeviceToHost);
	cuCheck(__LINE__);

	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);
	cuCheck(__LINE__);

	free(hostA);
	free(hostB);

	return hostC;
}
