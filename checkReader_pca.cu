#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#define YSIZE 600
#define XSIZE 1200
#define TrainDataNum 1934 //modified from 7736 to 1934
#define DataLen 1024
#define DataWidth 32
#define NumX 1010
#define NumWidth 128
#define NumY 53
#define NumHeight 64
#define Reduced_Data_Length 64
#define KNN_BLOCK_SIZE 32

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
int* labelIndexKernel=NULL;


__global__ void knn(float* trainDataKernel, float* distantKernel, int* labelIndexKernel, int count) {
	__shared__ float digit[Reduced_Data_Length];
	__shared__ float train[Reduced_Data_Length];
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	float cur;

	train[tx]=trainDataKernel[bx*Reduced_Data_Length+tx];
	train[KNN_BLOCK_SIZE+tx]=trainDataKernel[bx*Reduced_Data_Length+KNN_BLOCK_SIZE+tx];

	for(int i=0; i<count; ++i) {
		digit[tx]=testDigit[i*Reduced_Data_Length+tx];
		digit[KNN_BLOCK_SIZE+tx]=testDigit[i*Reduced_Data_Length+KNN_BLOCK_SIZE+tx];
		__syncthreads();

		cur=digit[tx]-train[tx];
		cur=cur*cur;
		digit[tx]=cur;
		cur=digit[KNN_BLOCK_SIZE+tx]-train[KNN_BLOCK_SIZE+tx];
		cur=cur*cur;
		digit[KNN_BLOCK_SIZE+tx]=cur;
		for(int stride=KNN_BLOCK_SIZE; stride>0; stride/=2) {
			__syncthreads();
			if(tx<stride)
				digit[tx]+=digit[tx+stride];
		}
		__syncthreads();
		if(tx==0) {
			distantKernel[TrainDataNum*i+bx]=digit[0];
			labelIndexKernel[TrainDataNum*i+bx]=bx;
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


//merge sort helper
void merge(float* distantHost, int n, int m) {
	int i, j, k;
	float* x=(float*)malloc(n*sizeof(float));
	int* y=(int*)malloc(n*sizeof(int));
	for(i=0, j=m, k=0; k<n; k++) {
		if(j==n) {
			x[k]=distantHost[i];
			y[k]=digits[i];
			i+=1;
		}else if(i==m) {
			x[k]=distantHost[j];
			y[k]=digits[j];
			j+=1;
		}else if(int(distantHost[j])<int(distantHost[i])) {
			x[k]=distantHost[j];
			y[k]=digits[j];
			j+=1;
		}else {
			x[k]=distantHost[i];
			y[k]=digits[i];
			i+=1;
		}
	}
	for(int i=0; i<n; i++) {
		distantHost[i]=x[i];
		digits[i]=y[i];
	}
	free(x);
	free(y);
}


//sort the output array from knn
void mergeSort(float* distantHost, int n) {
	if(n<2) {
		return;
	}
	int m=n/2;
	mergeSort(distantHost, m);
	mergeSort(distantHost+m, n-m);
	merge(distantHost, n, m);
}

void recognize(int* ans, int count) {
	int distantSize=sizeof(float)*TrainDataNum*15;
	dim3 dimBlock(DataWidth, DataWidth, 1);
	dim3 dimGrid(TrainDataNum, 1, 1);
	
	knn<<<dimGrid, dimBlock>>>(trainDataKernel, distantKernel, count);
	
	cudaDeviceSynchronize();
	cuCheck(__LINE__);
	cudaMemcpy(distantHost, distantKernel, distantSize, cudaMemcpyDeviceToHost);
	cuCheck(__LINE__);
	for(int j=0; j<count; ++j) {
		float* curDistantHost=distantHost+j*TrainDataNum;
		mergeSort(curDistantHost, TrainDataNum);
		int num[10]={};
		if(curDistantHost[0]>180) {
			ans[j]=-1;
			continue;
		}
		for(int i=0; i<12; i++) {
			num[digits[int(10000*(curDistantHost[i]-int(curDistantHost[i])))]]+=1;
		}
		int curBest=-1;
		int curInt=-1;
		for(int i=0; i<10; i++) {
			if(num[i]!=0&&num[i]>curBest) {
				curBest=num[i];
				curInt=i;
			}
		}
		ans[j]=curInt;
	}
}


__global__ void matrixMultiply_device(float *A, float *B, float *C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
	__shared__ float sharedA[32][32];
	__shared__ float sharedB[32][32];
	int Row=blockIdx.y*32+threadIdx.y;
	int Column=blockIdx.x*32+threadIdx.x;
	float resultC=0.0f;
	for (int i=0; i<ceil(numAColumns/32.0f); i++)
	{
		if (Row<numARows && i*32+threadIdx.x<numAColumns)
			sharedA[threadIdx.y][threadIdx.x]=A[Row*numAColumns+i*32+threadIdx.x];
		else
			sharedA[threadIdx.y][threadIdx.x]=0.0f;
		if (i*32+threadIdx.y<numBRows && Column<numBColumns)
			sharedB[threadIdx.y][threadIdx.x]=B[(i*32+threadIdx.y)*numBColumns+Column];
		else
			sharedB[threadIdx.y][threadIdx.x]=0.0f;
		__syncthreads();
		for (int j=0; j<32; j++)
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
	int numBColumns=Reduced_Data_Length;
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

	dim3 DimGrid(ceil(numCColumns/32.0f),ceil(numCRows/32.0f),1);
	dim3 DimBlock(32,32,1);
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
