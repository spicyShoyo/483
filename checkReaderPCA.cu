#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#define HISTOGRAM_LENGTH 256
#define YSIZE 600
#define XSIZE 1200
#define TrainDataNum 1934
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
#define Reduced_Data_Length 64
#define KNN_BLOCK_SIZE 32
#define TrainDigits "trainData/digits.csv"
#define TrainLabels "trainData/labels.csv"
#define PCATrainDigits "explore/trainingData.csv"
#define PCATrainLabels "explore/labelsPCA.csv"
#define PCAVec "explore/eigenvectors.csv"
#define K 4

void cuCheck(int line) {
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Error: %s, %d\n", cudaGetErrorString(err), line);
    }
}
//nvcc -arch sm_20 checkReader2.cu

// __constant__ float testDigit[DataLen*15];
__constant__ float testPCADigit[Reduced_Data_Length*15];
//image IO code below{{{{{{{{{{{{{{{{{{{{{{{{


void ppmReader(char* fileName, int** container, int* canvasHeight, int* canvasWidth) {
	int width=0;
	int height=0;
	FILE *ptr=fopen(fileName, "r");
	char* buffer=(char*)malloc(sizeof(char)*20);
	fscanf(ptr, "%s", buffer);
	fscanf(ptr, "%chandwriting dataset", buffer);
	while(getc(ptr)!=10) {};
	fscanf(ptr, "%s", buffer);
	width=(int)strtof(buffer, NULL);
	fscanf(ptr, "%c", buffer);
	fscanf(ptr, "%s", buffer);
	height=(int)strtof(buffer, NULL);
	fscanf(ptr, "%c", buffer);
	fscanf(ptr, "%c%c%c%c", buffer, buffer, buffer, buffer);
	*container=(int*)malloc(3*width*height*sizeof(int));
	for(int i=0; i<3*width*height; ++i) {
		fscanf(ptr, "%s", buffer);
		(*container)[i]=(int)(strtof(buffer, NULL));
		fscanf(ptr, "%c", buffer);
	}
	fclose(ptr);
	*canvasWidth=width;
	*canvasHeight=height;
	free(buffer);
}


//output black/white in gray scale
void monoWritterP2(int* image, int ySize, int xSize, char* fileName) {
	FILE *ptr=fopen(fileName, "w");
	fprintf(ptr, "P2\n");
	fprintf(ptr, "%d %d\n", xSize, ySize);
	fprintf(ptr, "255\n");
	for(int i=0; i<ySize*xSize; ++i) {
		fprintf(ptr, "%d\n", 255*(1-image[i]));
	}
	return;
}


void ppmWritter(char* fileName, int* container, int canvasHeight, int canvasWidth) {
	FILE *ptr=fopen(fileName, "w");
	fprintf(ptr, "P3\n");
	fprintf(ptr, "%d %d\n", canvasWidth, canvasHeight);
	fprintf(ptr, "255\n");
	for(int i=0; i<canvasHeight*canvasWidth*3; ++i) {
		fprintf(ptr, "%d\n", container[i]);
	}
	return;
}

//output black/white in black/white
void monoWritter(int* image, int ySize, int xSize, char* fileName) {
	FILE *ptr=fopen(fileName, "w");
	fprintf(ptr, "P1\n");
	fprintf(ptr, "%d %d\n", xSize, ySize);
	for(int i=0; i<ySize*xSize; ++i) {
		fprintf(ptr, "%d\n", image[i]);
	}
	return;
}


void outputImage(int* imageDevice, int ySize, int xSize, char* fileName="test.pgm") {
	int imageSize=sizeof(int)*ySize*xSize;
	int* imageHost=(int*)malloc(imageSize);
	cudaMemcpy(imageHost, imageDevice, imageSize, cudaMemcpyDeviceToHost);
	cuCheck(__LINE__);
	monoWritter(imageHost, ySize, xSize, fileName);
	free(imageHost);
	return;
}


void printFirstDigit(float* datas, int h=32, int w=32) {
	for(int i=0; i<h; i++) {
		for(int j=0; j<w; j++) {
			printf("%d", (int)datas[i*w+j]);
		}
		printf("\n");
	}
	printf("---------------\n");
	return;
}
//image IO code above}}}}}}}}}}}}}}}}}}}}}}}


//knn code below{{{{{{{{{{{{{{{{{{{{{{{{
int* labels=NULL;
float* eigenvectorsDevice;
float* trainDataPCAHost=NULL;
float* trainDataPCADevice=NULL;
float* distantPCAHost=NULL;
float* distantPCAKernel=NULL;

void freePCAKNN() {
	free(labels);
	cudaFree(eigenvectorsDevice);
	free(trainDataPCAHost);
	cudaFree(trainDataPCADevice);
	free(distantPCAHost);
	cudaFree(distantPCAKernel);
}
//init the answer to the training data
//return by pointer
//this is because the training data only holds the data
//it self, so here is what the data is.
void initLabels(int* container) {
	FILE* ptr=fopen(PCATrainLabels, "r");
	char* buffer=(char*)malloc(sizeof(char));
	for(int i=0; i<TrainDataNum; ++i) {
		fscanf(ptr, "%c", buffer);
		container[i]=(int)(buffer[0])-48;
		fscanf(ptr, "%c", buffer);
	}
	free(buffer);
	return;
}

void initEigenvectors() {
	float* eigenVectorsHost=(float*)malloc(DataLen*Reduced_Data_Length*sizeof(float));
	cudaMalloc((void **) &eigenvectorsDevice, DataLen*Reduced_Data_Length*sizeof(float));
	float* container=eigenVectorsHost;
	FILE* ptr=fopen(PCAVec, "r");
	char* buffer=(char*)malloc(sizeof(char));
	for(int j=0; j<DataLen; ++j) {
		for(int i=0; i<Reduced_Data_Length; ++i) {
			fscanf(ptr, "%f", &container[j*Reduced_Data_Length+i]);
			fscanf(ptr, "%c", buffer);
		}
	}
	free(buffer);
	cudaMemcpy(eigenvectorsDevice, eigenVectorsHost, DataLen*Reduced_Data_Length*sizeof(float), cudaMemcpyHostToDevice);
	cuCheck(__LINE__);
	free(eigenVectorsHost);
	return;
}


void initTrainDataPCAHost(float* container) {
	FILE* ptr=fopen(PCATrainDigits, "r");
	char* buffer=(char*)malloc(sizeof(char));
	for(int i=0; i<TrainDataNum; i++) { 
		for(int j=0; j<Reduced_Data_Length; j++) {
			fscanf(ptr, "%f", &container[i*Reduced_Data_Length+j]);
			fscanf(ptr, "%c", buffer);
		}
	}
	free(buffer);
	return;
}

void initPCAKNN() {
	printf("Initiallize...\n");
	labels=(int*)malloc(TrainDataNum*sizeof(int));
	initLabels(labels);
	cuCheck(__LINE__);
	int distantPCASize=sizeof(float)*TrainDataNum*15;
	distantPCAHost=(float*)malloc(distantPCASize);
	cudaMalloc((void**) &distantPCAKernel, distantPCASize);
	cuCheck(__LINE__);

	int trainDataPCASize=sizeof(float)*Reduced_Data_Length*TrainDataNum;
	trainDataPCAHost=(float*)malloc(trainDataPCASize);
	initTrainDataPCAHost(trainDataPCAHost);
	cudaMalloc((void**) &trainDataPCADevice, trainDataPCASize);
	cuCheck(__LINE__);
	cudaMemcpy(trainDataPCADevice, trainDataPCAHost, trainDataPCASize, cudaMemcpyHostToDevice);
	cuCheck(__LINE__);

	initEigenvectors();
	printf("Initiallize Done\n");
}


__global__ void knnPCA(float* trainDataKernel, float* distantKernel, int count) {
	__shared__ float digit[Reduced_Data_Length];
	__shared__ float train[Reduced_Data_Length];
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	float cur;

	train[tx]=trainDataKernel[bx*Reduced_Data_Length+tx];
	train[KNN_BLOCK_SIZE+tx]=trainDataKernel[bx*Reduced_Data_Length+KNN_BLOCK_SIZE+tx];

	for(int i=0; i<count; ++i) {
		digit[tx]=testPCADigit[i*Reduced_Data_Length+tx];
		digit[KNN_BLOCK_SIZE+tx]=testPCADigit[i*Reduced_Data_Length+KNN_BLOCK_SIZE+tx];
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
		}
		__syncthreads();
	}
}


void quick_sort (float *a, int* b, int n) {
    int i;
    int j;
    float p;
    float t;
    int t2;
    if (n < 2)
        return;
    p = a[n / 2];
    for (i = 0, j = n - 1;; i++, j--) {
        while (a[i] < p)
            i++;
        while (p < a[j])
            j--;
        if (i >= j)
            break;
        t = a[i];
        a[i] = a[j];
        a[j] = t;
        t2=b[i];
        b[i]=b[j];
        b[j]=t2;
    }
    quick_sort(a, b, i);
    quick_sort(a + i, b+i, n - i);
}


void recognizePCA(int* ans, int count) {
	int distantPCASize=sizeof(float)*TrainDataNum*count;
	int* curLabels=(int*)malloc(sizeof(int)*TrainDataNum);

	dim3 dimBlock(KNN_BLOCK_SIZE, 1, 1);
	dim3 dimGrid(TrainDataNum, 1, 1);
	knnPCA<<<dimGrid, dimBlock>>>(trainDataPCADevice, distantPCAKernel, count);

	cudaDeviceSynchronize();
	cuCheck(__LINE__);

	cudaMemcpy(distantPCAHost, distantPCAKernel, distantPCASize, cudaMemcpyDeviceToHost);
	cuCheck(__LINE__);

	for(int j=0; j<count; ++j) {
		memcpy(curLabels, labels, sizeof(int)*TrainDataNum);
		float* curDistantHost=distantPCAHost+j*TrainDataNum;
		quick_sort(curDistantHost, curLabels, TrainDataNum);
		int num[10]={};
		if(curDistantHost[0]>100) {
			ans[j]=-1;
			continue;
		}
		for(int i=0; i<K; i++) {
			num[curLabels[i]]+=1;
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


//strip elimination below{{{{{{{{{{{{{{{{{{{{{{{{
__global__ void stripEliminationDevice(int* checkMonoDevice, int* outDevice, int ySize=YSIZE, int xSize=XSIZE) {
	__shared__ int partialSum[1024];
	unsigned int tx=threadIdx.x;
	unsigned int ty=blockIdx.y;
	unsigned int start=2*blockIdx.x*blockDim.x;
	partialSum[tx]=0;
	if(start+tx<xSize) {
		partialSum[tx]=checkMonoDevice[ty*(xSize)+start+tx];
	}
	if(start+blockDim.x+tx<xSize) {
		partialSum[tx]+=checkMonoDevice[ty*(xSize)+start+blockDim.x+tx];
	}
	for(unsigned int stride=blockDim.x/2; stride>0; stride/=2) {
		__syncthreads();
		if(tx<stride) {
			partialSum[tx]+=partialSum[tx+stride];
		}
	}
	__syncthreads();
	if(tx==0) {
		outDevice[ty]=partialSum[tx];
	}
	if(partialSum[0]>500) {
		if(ty*(xSize)+start+tx<xSize*ySize) {
			checkMonoDevice[ty*(xSize)+start+tx]=0;
		}
		if(ty*(xSize)+start+blockDim.x+tx<xSize*ySize) {
			checkMonoDevice[ty*(xSize)+start+blockDim.x+tx]=0;
		}
	}
}


void stripEliminationHost(int* checkMonoDevice, int ySize, int xSize) {
	int* outHost;
	int* outDevice;
	int outSize=sizeof(int)*ySize;
	outHost=(int*)malloc(outSize);
	cudaMalloc((void **) &outDevice, outSize);

	dim3 dimBlock(1024, 1, 1);
	dim3 dimGrid(1, ySize, 1);
	stripEliminationDevice<<<dimGrid, dimBlock>>>(checkMonoDevice, outDevice, ySize, xSize);
	cudaDeviceSynchronize();
	cuCheck(__LINE__);

	cudaMemcpy(outHost, outDevice, outSize, cudaMemcpyDeviceToHost);
	free(outHost);
	cudaFree(outDevice);

	return;
}
//strip elimination code above}}}}}}}}}}}}}}}}}}}}}}}


//scale code below{{{{{{{{{{{{{{{{{{{{{{{{
void scaleHost(int** checkColoredDevice, int* image, int ySize, int xSize) {
	//alloc memory
	int checkColoredSize=sizeof(int)*3*ySize*xSize;
	cudaMalloc((void **) checkColoredDevice, checkColoredSize);
	cuCheck(__LINE__);

	//scale, assume no need to scale now
	cudaMemcpy(*checkColoredDevice, image, checkColoredSize, cudaMemcpyHostToDevice);
	cuCheck(__LINE__);

	return;
}
//scale code above}}}}}}}}}}}}}}}}}}}}}}}


//verification cuda below{{{{{{{{{{{{{{{{{{{{{{{{

__global__ void verifyBlue(int* input_arr, int* isBlue) {
	int col=blockIdx.x*blockDim.x+threadIdx.x;
	int row=blockIdx.y*blockDim.y+threadIdx.y;
	int input_index=row*XSIZE+col;
	int output_index=row*2048+col;
	if (row<YSIZE && col<XSIZE && input_arr[3*input_index]<RED_THRESHOLD && input_arr[3*input_index+1]<GREEN_THRESHOLD && input_arr[3*input_index+2]>BLUE_THRESHOLD)
		isBlue[output_index]=1;
	else
		isBlue[output_index]=0;
	return;
}

__global__ void countBlue_perRow(int* isBlue, int* numBlue_perRow) {
	__shared__ int partial[2048];
	int BLOCK_SIZE=1024;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	partial[tx]=isBlue[by*2*BLOCK_SIZE+tx];
	partial[BLOCK_SIZE+tx]=isBlue[by*2*BLOCK_SIZE+BLOCK_SIZE+tx];
	__syncthreads();
	for (int stride=BLOCK_SIZE; stride>0; stride/=2)
	{
		if (tx<stride)
			partial[tx]+=partial[tx+stride];
		__syncthreads();
	}
	if (tx==by)
		numBlue_perRow[tx]=partial[0];
	return;
}

__global__ void countBlue_total(int* numBlue_perRow, int* total) {
	__shared__ int partial[1024];
	int BLOCK_SIZE=512;
	int bx=blockIdx.x;
	int tx=threadIdx.x;
	partial[tx]=numBlue_perRow[bx*2*BLOCK_SIZE+tx];
	partial[BLOCK_SIZE+tx]=numBlue_perRow[bx*2*BLOCK_SIZE+BLOCK_SIZE+tx];
	__syncthreads();
	for (int stride=BLOCK_SIZE; stride>0; stride/=2)
	{
		if (tx<stride)
			partial[tx]+=partial[tx+stride];
		__syncthreads();
	}
	if (tx==bx)
		total[tx]=partial[0];
	return;
}

int verificationHost(int* checkColoredDevice, int ySize=YSIZE, int xSize=XSIZE) {
	int* device_check;
	int* isBlue;
	int* numBlue_perRow;
	int* host_total;
	int* device_total;
	int size=ySize*xSize;

	cudaMalloc((void**)&device_check, 3*size*sizeof(int));
	cudaMalloc((void**)&isBlue, 2048*1024*sizeof(int));

	cudaMemcpy(device_check, checkColoredDevice, 3*size*sizeof(int), cudaMemcpyHostToDevice);
	cuCheck(__LINE__);

	dim3 DimGrid1(2, 1024, 1);
	dim3 DimBlock1(1024, 1, 1);
	verifyBlue<<<DimGrid1, DimBlock1>>>(device_check, isBlue);
	cudaFree(device_check);
	cuCheck(__LINE__);

	cudaMalloc((void**)&numBlue_perRow, 1024*sizeof(int));
	host_total=(int*)malloc(sizeof(int));
	cudaMalloc((void**)&device_total, sizeof(int));
	cuCheck(__LINE__);

	dim3 DimGrid2(1, 1024, 1);
	dim3 DimBlock2(1024, 1, 1);
	countBlue_perRow<<<DimGrid2, DimBlock2>>>(isBlue, numBlue_perRow);
	cudaFree(isBlue);
	cuCheck(__LINE__);

	dim3 DimGrid3(1, 1, 1);
	dim3 DimBlock3(512, 1, 1);
	countBlue_total<<<DimGrid3, DimBlock3>>>(numBlue_perRow, device_total);
	cudaFree(numBlue_perRow);
	cuCheck(__LINE__);

	cudaMemcpy(host_total, device_total, sizeof(int), cudaMemcpyDeviceToHost);
	cuCheck(__LINE__);

	int total=*host_total;

	free(host_total);
	cudaFree(device_total);

	//printf("number of blue pixels are %d\n", total);

	if (total>size/8 && total<size/2)
		return 1;
	return 0;
}
//verification code above}}}}}}}}}}}}}}}}}}}}}}}


//convert to mono code below{{{{{{{{{{{{{{{{{{{{{{{{
__global__ void toMonoDevice(int* checkMonoDevice, int* checkColoredDevice, int ySize, int xSize) {
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<ySize*xSize) {
		int r=checkColoredDevice[3*index];
		int g=checkColoredDevice[3*index+1];
		int b=checkColoredDevice[3*index+2];
		checkMonoDevice[index]=((r+g+b)/3)>100? 0:1;
	}
}


void toMonoHost(int** checkMonoDevice, int* checkColoredDevice, int ySize=YSIZE, int xSize=XSIZE) {
	//alloc memory
	int checkMonoSize=sizeof(int)*ySize*xSize;
	cudaMalloc((void **) checkMonoDevice, checkMonoSize);
	cuCheck(__LINE__);

	//convert to gray here
	dim3 dimBlock(1024, 1, 1);
	dim3 dimGrid(ceil(xSize*ySize/(float)1024), 1, 1);
	toMonoDevice<<<dimGrid, dimBlock>>>(*checkMonoDevice, checkColoredDevice, ySize, xSize);
	cudaDeviceSynchronize();
	cuCheck(__LINE__);

	// outputImage(*checkMonoDevice, ySize, xSize, "checkMono.pgm");

	return;
}
//convert to mono code above}}}}}}}}}}}}}}}}}}}}}}}


//grab and read digit code below{{{{{{{{{{{{{{{{{{{{{{{{
__global__ void getHorizonDevice(int* horizonDevice, int* checkMonoDevice, int grabX, int grabY, int grabWidth, int grabHeight, int ySize=YSIZE, int xSize=XSIZE) {
	int bx=blockIdx.x;
	int inX=bx+grabX;
	int inY=grabY;
	int ans=0;
	for(int i=0; i<grabHeight; ++i) {
		ans+=checkMonoDevice[(i+inY)*xSize+inX];
	}
	horizonDevice[bx]=ans;
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


void setPCAConstant(float* digitHost, int count) {
	int digitSize=sizeof(float)*count*DataLen;
	int digitPCADSize=sizeof(float)*count*Reduced_Data_Length;
	float* digitDevice;
	float* digitPCADevice;
	cudaMalloc((void **) &digitDevice, digitSize);
	cudaMalloc((void **) &digitPCADevice, digitPCADSize);
	cudaMemcpy(digitDevice, digitHost, digitSize, cudaMemcpyHostToDevice);
	cuCheck(__LINE__);

	dim3 dimGrid(ceil(Reduced_Data_Length/32.0f),ceil(count/32.0f),1);
	dim3 dimBlock(32,32,1);
	matrixMultiply_device<<<dimGrid, dimBlock>>>(digitDevice, eigenvectorsDevice, digitPCADevice, count, DataLen, DataLen, Reduced_Data_Length, count, Reduced_Data_Length);
	cudaDeviceSynchronize();
	cuCheck(__LINE__);

	cudaMemcpyToSymbol(testPCADigit, digitPCADevice, Reduced_Data_Length*count*sizeof(float));

	cuCheck(__LINE__);
	cudaFree(digitDevice);
	cudaFree(digitPCADevice);
}


int* readAreaHost(int* checkMonoDevice, int grabX, int grabY, int grabWidth, int grabHeight, int num, int ySize=YSIZE, int xSize=XSIZE) {
	//outputImage(checkMonoDevice, ySize, xSize, "area.pgm");
	int count=0;
	int checkMonoSize=sizeof(int)*ySize*xSize;
	int* checkMonoHost=(int*)malloc(checkMonoSize);
	cudaMemcpy(checkMonoHost, checkMonoDevice, checkMonoSize, cudaMemcpyDeviceToHost);
	cuCheck(__LINE__);
	//15 is the length of routing number + account number
	int* ans=(int*)malloc(sizeof(int)*15);
	//set all to -1 as init state
	for(int i=0; i<15; ++i) {
		ans[i]=-1;
	}

	int horizonSize=sizeof(int)*grabWidth;
	int* horizonHost;
	int* horizonDevice;
	horizonHost=(int*)malloc(horizonSize);
	cudaMalloc((void **) &horizonDevice, horizonSize);

	dim3 dimBlock(1, 1, 1);
	dim3 dimGrid(grabWidth, 1, 1);
	getHorizonDevice<<<dimGrid, dimBlock>>>(horizonDevice, checkMonoDevice, grabX, grabY, grabWidth, grabHeight, ySize, xSize);
	cudaDeviceSynchronize();
	cuCheck(__LINE__);
	//outputImage(horizonDevice, 1, grabWidth);
	cudaMemcpy(horizonHost, horizonDevice, horizonSize, cudaMemcpyDeviceToHost);
	cudaFree(horizonDevice);
	cuCheck(__LINE__);
	float* digitHost=(float*)malloc(sizeof(float)*DataLen*15);
	// int testDigitSize=sizeof(float)*DataLen;
	int left=0;
	int right=0;
	while(left<grabWidth&&count<num) {
		if(horizonHost[left]==0) {
			++left;
		}else {
			right=left;
			while(right<grabWidth&&horizonHost[right]!=0) {
				++right;
			}

			int midX=(left+right)/2+grabX;
			int top=-1;
			int bottom=-1;
			//printf("%d, %d\n", left, right);
			for(int y=0; (y<grabHeight/2&&((top==-1)||(bottom==-1))); ++y) {
				int yt=y+grabY;
				int yb=grabY+grabHeight-y;
				for(int xi=left+grabX; xi<right+grabX; ++xi) {
					if(top==-1&&checkMonoHost[yt*xSize+xi]!=0) {
						top=yt;
					}
					if(bottom==-1&&checkMonoHost[yb*xSize+xi]!=0) {
						bottom=yb;
					}
				}
			}
			int midY=(top+bottom)/2;
			int startX=midX-16;
			int startY=midY-16;
			for(int i=startY; i<startY+DataWidth; ++i) {
				for(int j=startX; j<startX+DataWidth; ++j) {
					int x=j-startX;
					int y=i-startY;
					//printf("%d", checkMonoHost[i*xSize+j]);
					digitHost[count*DataLen+y*DataWidth+x]=(float)checkMonoHost[i*xSize+j];
				}
				//printf("\n");
			}
			++count;
			left=right;
		}

	}

	setPCAConstant(digitHost, count);
	recognizePCA(ans, count);

	free(digitHost);
	free(checkMonoHost);	
	return ans;
}
//grab and read digit code above}}}}}}}}}}}}}}}}}}}}}}}


//read check code below{{{{{{{{{{{{{{{{{{{{{{{{
void checkReaderHost(int* checkMonoDevice, int ySize=YSIZE, int xSize=XSIZE) {
	stripEliminationHost(checkMonoDevice, ySize, xSize);

	//grab the area left, but later I found that don't really need to.
	int* ans;

	ans=readAreaHost(checkMonoDevice, RoutingX, RoutingY, RoutingWidth, RoutingHeight, 15, ySize, xSize);
	
	char routing[9];
	for(int i=0; i<9; ++i) {
		routing[i]=char(ans[i]+48);
	}
	char account[6];
	for(int i=9; i<15; ++i) {
		account[i-9]=char(ans[i]+48);
	}
	free(ans);
	printf("Routing Number: %s\n", routing);
	printf("Account Number: %s\n", account);

	ans=readAreaHost(checkMonoDevice, MoneyX, MoneyY, MoneyWidth, MoneyHeight, 6, ySize, xSize);
	float money=0;
	int j=0;
	while(ans[j]!=-1) {
		money=money*10+ans[j];
		j++;
	}
	money=money/100;
	free(ans);
	printf("Amount: %.2f\n", money);

	char num[3];
	ans=readAreaHost(checkMonoDevice, NumX, NumY, NumWidth, NumHeight, 3, ySize, xSize);
	for(int i=0; i<3; ++i) {
		num[i]=char(ans[i]+48);
	}
	free(ans);
	printf("Check Number: %s\n", num);

	return;
}
//read check code above}}}}}}}}}}}}}}}}}}}}}}}


void readSingleCheck(int* in) {
	//read in the image
	int* imageHost=NULL;
	int* ySize=(int*)malloc(sizeof(int));
	int* xSize=(int*)malloc(sizeof(int));

	int* checkColoredDevice=NULL;
	
	*xSize=1200;
	*ySize=600;
	imageHost=(int*)malloc(sizeof(int));
	checkColoredDevice=in;

	//verify check
	int valid=verificationHost(checkColoredDevice);
	if(!valid){
	 	printf("Invalid check\n");
	 	return;
	}
	else
	 	printf("Valid Check from Chase Bank\n");

	//convert to mono
	int* checkMonoDevice=NULL;
	toMonoHost(&checkMonoDevice, checkColoredDevice);

	//read the check
	checkReaderHost(checkMonoDevice);

	//free memory
	free(imageHost);
	free(ySize);
	free(xSize);
	cudaFree(checkColoredDevice);
	cudaFree(checkMonoDevice);
	printf("\n");
	return;
}

bool isBlack_pixel(int x, int y, int *image_container, int width){
	int index=3*(y*width+x);
	int red=image_container[index];
	int green=image_container[index+1];
	int blue=image_container[index+2];
	return (red<BLACK_THRESHOLD && green<BLACK_THRESHOLD && blue<BLACK_THRESHOLD);
}

bool isBlack(int* container, int x, int y, int ySize, int xSize) {
	int sum=0;
	for(int i=-1; i<2; i+=1) {
		for(int j=-1; j<2; j+=1) {
			int xi=x+i;
			int yi=y+j;
			if(xi>=0 && xi<xSize && yi>=0 && yi<ySize)
				if(isBlack_pixel(xi, yi, container, xSize))
					sum+=1;
		}
	}
	if(sum>4) return true;
	return false;
}

bool width_greater_than_height(int *upperleft, int *upperright, int *lowerleft, int *lowerright){
	int curwidth=(upperleft[0]-upperright[0])*(upperleft[0]-upperright[0])+(upperleft[1]-upperright[1])*(upperleft[1]-upperright[1]);
	int curheight=(upperleft[0]-lowerleft[0])*(upperleft[0]-lowerleft[0])+(upperleft[1]-lowerleft[1])*(upperleft[1]-lowerleft[1]);
	return (curwidth>curheight);
}

int getNewX(int x, int y, int dir){
	if (dir==0) return x+1;
	else if (dir==2) return x-1;
	else return x;
}

int getNewY(int x, int y, int dir){
	if (dir==1) return y+1;
	else if (dir==3) return y-1;
	else return y;
}

void enqueue(int * queue, int value, int * head, int * tail, int maxSize)
{
    if ((*tail - maxSize) == *head)
    {
        printf("Queue is full\n");
        return;
    }
    *tail = *tail + 1;
    queue[*tail % maxSize] = value;
}

int dequeue(int * queue, int * head, int * tail, int maxSize)
{
    if (*head == *tail)
    {
        printf("Queue is empty\n");
        return -1;
    }
    *head = *head + 1;
    return queue[*head % maxSize];
}


void bfs(int *image_container, int *upperleft, int *upperright, int *lowerleft, int *lowerright, int *check_width, int *check_height, int width, int height){
	upperleft[0]=width-1;
	upperleft[1]=height-1;
	upperright[0]=0;
	upperright[1]=height-1;
	lowerleft[0]=width-1;
	lowerleft[1]=0;
	lowerright[0]=0;
	lowerright[1]=0;
	int cur_coord_y=0;
	while (!isBlack(image_container, width/2, cur_coord_y, height, width))
		cur_coord_y+=1;
    int head_x=0;
    int tail_x=0;
    int head_y=0;
    int tail_y=0;
    int queueSize=width*height;
    int* coordinates_x=(int*)malloc(queueSize*sizeof(int));
    int* coordinates_y=(int*)malloc(queueSize*sizeof(int));
    int* visited=(int*)malloc(queueSize*sizeof(int));
    int i;
    int j;
    for (i=0; i<height; i++)
    	for (j=0; j<width; j++)
    		visited[i*width+j]=0;
    enqueue(coordinates_x, width/2, &head_x, &tail_x, queueSize);
    enqueue(coordinates_y, cur_coord_y, &head_y, &tail_y, queueSize);
	while (!(head_x==tail_x && head_y==tail_y))
	{
		int X=dequeue(coordinates_x, &head_x, &tail_x, queueSize);
		int Y=dequeue(coordinates_y, &head_y, &tail_y, queueSize);
		if (X<upperleft[0]){
			upperleft[0]=X;
			upperleft[1]=Y;
		}
		if (X>lowerright[0]){
			lowerright[0]=X;
			lowerright[1]=Y;
		}
		if (Y<upperright[1]){
			upperright[0]=X;
			upperright[1]=Y;
		}
		if (Y>lowerleft[1]){
			lowerleft[0]=X;
			lowerleft[1]=Y;
		} 
		for (int dir=0; dir<4; dir++)
		{
			int newX=getNewX(X, Y, dir);
			int newY=getNewY(X, Y, dir);
			if (newX>=0 && newX<width && newY>=0 && newY<height && isBlack(image_container, newX, newY, height, width))
			{
				if (visited[newY*width+newX]==0)
				{
					visited[newY*width+newX]=1;
					enqueue(coordinates_x, newX, &head_x, &tail_x, queueSize);
					enqueue(coordinates_y, newY, &head_y, &tail_y, queueSize);
				}
			}
		}
	}
	free(coordinates_x);
	free(coordinates_y);
	free(visited);

	if (!width_greater_than_height(upperleft, upperright, lowerleft, lowerright)){ //to be tested
		int x=upperleft[0];
		int y=upperleft[1];
		upperleft[0]=upperright[0];
		upperleft[1]=upperright[1];
		upperright[0]=lowerright[0];
		upperright[1]=lowerright[1];
		lowerright[0]=lowerleft[0];
		lowerright[1]=lowerleft[1];
		lowerleft[0]=x;
		lowerleft[1]=y;
	}
	*check_width=sqrt((upperleft[0]-upperright[0])*(upperleft[0]-upperright[0])+(upperleft[1]-upperright[1])*(upperleft[1]-upperright[1]));
	*check_height=sqrt((upperleft[0]-lowerleft[0])*(upperleft[0]-lowerleft[0])+(upperleft[1]-lowerleft[1])*(upperleft[1]-lowerleft[1]));
	*check_width=*check_width/(float)(XSIZE-2*Boundary)*XSIZE;
	*check_height=*check_height/(float)(YSIZE-2*Boundary)*YSIZE;
}

__global__ void rotation(int* device_input, int* device_output, int input_height, int input_width, int output_height, int output_width, int center_x, int center_y, float cos_theta, float sin_theta){
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int dx=blockDim.x;
	int dy=blockDim.y;
	int output_coord_x=bx*dx+tx;
	int output_coord_y=by*dy+ty;
	int normal_output_x=output_coord_x-output_width/2;
	int normal_output_y=output_coord_y-output_height/2;
	int output_index=output_coord_y*output_width+output_coord_x;
	if (output_coord_x<output_width && output_coord_y<output_height){
		int normal_input_x=cos_theta*normal_output_x+sin_theta*normal_output_y;
		int normal_input_y=cos_theta*normal_output_y-sin_theta*normal_output_x;
		int input_coord_x=normal_input_x+center_x;
		int input_coord_y=normal_input_y+center_y;
		int input_index=input_coord_y*input_width+input_coord_x;
		device_output[3*output_index]=device_input[3*input_index];
		device_output[3*output_index+1]=device_input[3*input_index+1];
		device_output[3*output_index+2]=device_input[3*input_index+2];
	}
}


float getCos(int x1, int y1, int x2, int y2){
	int a=x2-x1;
	int b=y1-y2;
	return (float)a/sqrt((float)(a*a+b*b));
}


float getSin(int x1, int y1, int x2, int y2){
	int a=x2-x1;
	int b=y1-y2;
	return (float)b/sqrt((float)(a*a+b*b));
}


__global__ void resize(int* input, int* output, int input_height, int input_width, int output_height, int output_width){
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int dx=blockDim.x;
	int dy=blockDim.y;
	int output_coord_x=bx*dx+tx;
	int output_coord_y=by*dy+ty;
	int output_index=output_coord_y*output_width+output_coord_x;
	if (output_coord_x<output_width && output_coord_y<output_height){
		int input_coord_x=output_coord_x*input_width/output_width;
		int input_coord_y=output_coord_y*input_height/output_height;
		int input_index=input_coord_y*input_width+input_coord_x;
		output[3*output_index]=input[3*input_index];
		output[3*output_index+1]=input[3*input_index+1];
		output[3*output_index+2]=input[3*input_index+2];
	}
}


int* preprocess(char* fileName){
	printf("\nCheck: %s\n", fileName);
	printf("BFS...\n");
	int check_width=0;
	int check_height=0;
	int upperleft[2];
	int upperright[2];
	int lowerleft[2];
	int lowerright[2];
	int height=0;
	int width=0;
	int* input_image;
	int* output_resized_image;

	ppmReader(fileName, &input_image, &height, &width);

	bfs(input_image, upperleft, upperright, lowerleft, lowerright, &check_width, &check_height, width, height);
	printf("BFS Done\n");
	// printf("upperleft= %d, %d\n", upperleft[0], upperleft[1]);
	// printf("upperright= %d, %d\n", upperright[0], upperright[1]);
	// printf("lowerleft= %d, %d\n", lowerleft[0], lowerleft[1]);
	// printf("lowerright= %d, %d\n", lowerright[0], lowerright[1]);
	// printf("check_width= %d\n", check_width);
	// printf("check_height= %d\n", check_height);

	int center_x=(upperleft[0]+lowerright[0]+upperright[0]+lowerleft[0])/4;
	int center_y=(upperleft[1]+lowerright[1]+upperright[1]+lowerleft[1])/4;

	float cos_theta=getCos(upperleft[0], upperleft[1], upperright[0], upperright[1]);
	float sin_theta=getSin(upperleft[0], upperleft[1], upperright[0], upperright[1]);

	int* device_input;
	int* device_raw_output;
	int* device_resized_output;
	int input_size=3*width*height*sizeof(int);
	int raw_output_size=3*check_height*check_width*sizeof(int);
	int resized_output_size=3*XSIZE*YSIZE*sizeof(int);

	output_resized_image=(int*)malloc(resized_output_size);
	cudaMalloc((void**)&device_input, input_size);
	cudaMalloc((void**)&device_raw_output, raw_output_size);
	cudaMalloc((void**)&device_resized_output, resized_output_size);
	cudaMemcpy(device_input, input_image, input_size, cudaMemcpyHostToDevice);
	cuCheck(__LINE__);

	dim3 dimBlock1(ceil(check_width/32.0), ceil(check_height/32.0), 1);
	dim3 dimGrid1(32, 32, 1);
	rotation<<<dimGrid1, dimBlock1>>>(device_input, device_raw_output, height, width, check_height, check_width, center_x, center_y, cos_theta, sin_theta);
	cuCheck(__LINE__);

	cudaDeviceSynchronize();

	dim3 dimBlock2(ceil(XSIZE/32.0), ceil(YSIZE/32.0), 1);
	dim3 dimGrid2(32, 32, 1);
	resize<<<dimGrid2, dimBlock2>>>(device_raw_output, device_resized_output, check_height, check_width, YSIZE, XSIZE);
	cuCheck(__LINE__);
	cudaDeviceSynchronize();

	cudaMemcpy(output_resized_image, device_resized_output, resized_output_size, cudaMemcpyDeviceToHost);

	cudaFree(device_input);
	cudaFree(device_raw_output);
	cuCheck(__LINE__);

	//ppmWritter("output.pgm", output_resized_image, YSIZE, XSIZE);
	free(input_image);
	free(output_resized_image);
	return device_resized_output;
}


int main() {
	initPCAKNN();
	readSingleCheck(preprocess("testCases/check1.ppm"));
	readSingleCheck(preprocess("testCases/check2.ppm"));
	readSingleCheck(preprocess("testCases/check3.ppm"));
	readSingleCheck(preprocess("testCases/check4.ppm"));
	readSingleCheck(preprocess("testCases/check5.ppm"));
	readSingleCheck(preprocess("testCases/check6.ppm"));
	readSingleCheck(preprocess("testCases/check7.ppm"));
	readSingleCheck(preprocess("testCases/check8.ppm"));
	freePCAKNN();
	return 0;
}

