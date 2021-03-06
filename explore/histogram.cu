#include "stdlib.h"
#include "stdio.h"
#define BlockWidth 32
#define HISTOGRAM_LENGTH 256

void cuCheck(int line) {
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Error: %s, %d\n", cudaGetErrorString(err), line);
    }
}
//nvcc -arch sm_20 histogram.cu


void ppmReader(float** container, int* canvasHeight, int* canvasWidth) {
	int width=0;
	int height=0;
	FILE *ptr=fopen("check10.ppm", "r");
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
	*container=(float*)malloc(3*width*height*sizeof(float));
	for(int i=0; i<3*width*height; ++i) {
		fscanf(ptr, "%s", buffer);
		(*container)[i]=strtof(buffer, NULL);
		fscanf(ptr, "%c", buffer);
	}
	fclose(ptr);
	*canvasWidth=width;
	*canvasHeight=height;
	free(buffer);
}


void grayWritter(unsigned char* image, int ySize, int xSize) {
	FILE *ptr=fopen("imageGray.pgm", "w");
	fprintf(ptr, "P2\n");
	fprintf(ptr, "%d %d\n", xSize, ySize);
	fprintf(ptr, "255\n");
	for(int i=0; i<ySize*xSize; ++i) {
		fprintf(ptr, "%d\n", (int)image[i]);
	}
	return;
}


void charWritter(unsigned char* image, int ySize, int xSize, int numChannel) {
	FILE *ptr=fopen("imageChar.pgm", "w");
	fprintf(ptr, "P3\n");
	fprintf(ptr, "%d %d\n", xSize, ySize);
	fprintf(ptr, "255\n");
	for(int i=0; i<ySize*xSize*numChannel; ++i) {
		fprintf(ptr, "%d\n", (int)image[i]);
	}
	return;
}

void floatWritter(float* image, int ySize, int xSize, int numChannel) {
	FILE *ptr=fopen("imageCorrect.pgm", "w");
	fprintf(ptr, "P3\n");
	fprintf(ptr, "%d %d\n", xSize, ySize);
	fprintf(ptr, "255\n");
	for(int i=0; i<ySize*xSize*numChannel; ++i) {
		fprintf(ptr, "%d\n", (int)image[i]);
	}
	return;
}


__global__ void f2ucDevice(unsigned char* imageCharDevice, float* imageFloatDevice, int ySize, int xSize, int numChannel) {
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<ySize*xSize*numChannel) {
		imageCharDevice[index]=(unsigned char)(imageFloatDevice[index]);
	}
}


void f2ucHost(unsigned char* imageCharDevice, float* imageFloatDevice, int ySize, int xSize, int numChannel) {
	dim3 dimBlock(1024, 1, 1);
	dim3 dimGrid(ceil(xSize*ySize*numChannel/(float)1024), 1, 1);
	f2ucDevice<<<dimGrid, dimBlock>>>(imageCharDevice, imageFloatDevice, ySize, xSize, numChannel);
	cudaDeviceSynchronize();
	cuCheck(__LINE__);
	return;
}


__global__ void uc2grDevice(unsigned char* imageGrayDevice, unsigned char* imageCharDevice, int ySize, int xSize, int numChannel) {
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<ySize*xSize) {
		int r=(int)imageCharDevice[3*index];
		int g=(int)imageCharDevice[3*index+1];
		int b=(int)imageCharDevice[3*index+2];
		//imageGrayDevice[index]=(unsigned char)index;
		float ri=0.21;
		float gi=0.71;
		float bi=0.07;
		imageGrayDevice[index]=(unsigned char)(int)(ri*r+gi*g+bi*b);
	}
}


void uc2grHost(unsigned char* imageGrayDevice, unsigned char* imageCharDevice, int ySize, int xSize, int numChannel) {
	dim3 dimBlock(1024, 1, 1);
	dim3 dimGrid(ceil(xSize*ySize/(float)1024), 1, 1);
	uc2grDevice<<<dimGrid, dimBlock>>>(imageGrayDevice, imageCharDevice, ySize, xSize, numChannel);
	cudaDeviceSynchronize();
	cuCheck(__LINE__);
	return;
}


__global__ void getHistDevice(int* histDevice, unsigned char* imageGrayDevice, int ySize, int xSize) {
	__shared__ int histPrivate[HISTOGRAM_LENGTH];
	if(threadIdx.x<HISTOGRAM_LENGTH) {
		histPrivate[threadIdx.x]=0;
	}
	__syncthreads();
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	int stride=blockDim.x*gridDim.x;
	while(i<ySize*xSize) {
		atomicAdd(&(histPrivate[imageGrayDevice[i]]), 1);
		i+=stride;
	}
	__syncthreads();
	if(threadIdx.x<HISTOGRAM_LENGTH) {
		atomicAdd(&(histDevice[threadIdx.x]), histPrivate[threadIdx.x]);
	}
}


void getHistHost(int* histDevice, unsigned char* imageGrayDevice, int ySize, int xSize) {
	dim3 dimBlock(256, 1, 1);
	dim3 dimGrid(ceil(ySize*xSize/(float)256), 1, 1);
	getHistDevice<<<dimGrid, dimBlock>>>(histDevice, imageGrayDevice, ySize, xSize);
	cudaDeviceSynchronize();
	cuCheck(__LINE__);
	return;
}


__global__ void scan(int* input, int* output, int* sum, int len) {
  int tx=threadIdx.x;
  int start=2*blockIdx.x*BlockWidth+threadIdx.x;
  __shared__ int sharedM[2*BlockWidth];
  if(start<len) {
    sharedM[tx]=input[start];
  }else {
    sharedM[tx]=0;
  }
  if(start+BlockWidth<len) {
    sharedM[tx+BlockWidth]=input[start+BlockWidth];
  }else {
    sharedM[tx+BlockWidth]=0;
  }
  __syncthreads();
  for(int stride=1; stride<=BlockWidth; stride*=2) {
    int index=(threadIdx.x+1)*stride*2-1;
    if(index<2*BlockWidth) {
      sharedM[index]+=sharedM[index-stride];
    }
    __syncthreads();
  }
  for(int stride=BlockWidth/2; stride>0; stride/=2) {
    __syncthreads();
    int index=(threadIdx.x+1)*stride*2-1;
    if(index+stride<2*BlockWidth) {
      sharedM[index+stride]+=sharedM[index];
    }
  }
  __syncthreads();
  if(start<len) {
    output[start]=sharedM[tx];
  }
  if(start+BlockWidth<len) {
    output[start+BlockWidth]=sharedM[tx+BlockWidth];
  }
  if(tx==0) {
    sum[blockIdx.x]=sharedM[2*BlockWidth-1];
  }
} 


__global__ void add(int* output, int* sum, int len) {
  int start=2*blockIdx.x*BlockWidth+threadIdx.x;
  for(int block=blockIdx.x; block>0; --block) {
    if(start<len) {
      output[start]+=sum[block-1];
    }
    if(start+BlockWidth<len) {
      output[start+BlockWidth]+=sum[block-1];
    }
  }
}


void getHistCDFHost(int* histCDFDevice, int* histDevice, int ySize, int xSize) {
	int* sumHost;
	int* sumDevice;
	int sumSize=sizeof(int)*ceil(HISTOGRAM_LENGTH/(float)(BlockWidth*2));
	sumHost=(int*)malloc(sumSize);
	cudaMalloc((void **) &sumDevice, sumSize);
	for(int i=0; i<ceil(HISTOGRAM_LENGTH/(float)(BlockWidth*2)); ++i) {
		sumHost[i]=0;
	}
	cudaMemcpy(sumDevice, sumHost, sumSize, cudaMemcpyHostToDevice);
	cuCheck(__LINE__);
	dim3 dimBlock(BlockWidth, 1, 1);
	dim3 dimGrid(ceil(HISTOGRAM_LENGTH/(float)(BlockWidth*2)), 1, 1);
	scan<<<dimGrid, dimBlock>>>(histDevice, histCDFDevice, sumDevice, ySize*xSize);
	cudaDeviceSynchronize();
	cuCheck(__LINE__);
	add<<<dimGrid, dimBlock>>>(histCDFDevice, sumDevice, HISTOGRAM_LENGTH);
	cudaMemcpy(sumHost, sumDevice, sumSize, cudaMemcpyDeviceToHost);
	free(sumHost);
	cudaFree(sumDevice);
	return;
}


__global__ void clampDevice(int* histClampDevice, int cdfmin) {
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<HISTOGRAM_LENGTH) {
		int cur=histClampDevice[index];
		int x=255*(cur-cdfmin)/(1-cdfmin);
		//return min(max(x, start), end)
		int maxPart=x>0? x:0;
		int minPart=maxPart<255? maxPart:255;
		histClampDevice[index]=minPart;
	}
}


void clampHost(int* histClampDevice, int cdfmin) {
	dim3 dimBlock(1, 1, 1);
	dim3 dimGrid(HISTOGRAM_LENGTH, 1, 1);
	clampDevice<<<dimGrid, dimBlock>>>(histClampDevice, cdfmin);
	cudaDeviceSynchronize();
	cuCheck(__LINE__);
	return;
}


int clamp(int x, int start, int end) {
	return min(max(x, start), end);
}


void cdf2clamp(float* histCDFHost, int* histClampHost) {
	int cdfmin=histCDFHost[0];
	for(int i=0; i<HISTOGRAM_LENGTH; ++i) {
		int x=255*(histCDFHost[i]-cdfmin)/(1-cdfmin);
		histClampHost[i]=clamp(x, 0, 255);
	}
}


float p(int x, int ySize, int xSize) {
	return x/(float)(ySize*xSize);
}


void getCDF(float* histCDFHost, int* histHost, int ySize, int xSize) {
	histCDFHost[0]=p(histHost[0], ySize, xSize);
	for(int i=1; i<HISTOGRAM_LENGTH; ++i) {
		histCDFHost[i]=histCDFHost[i-1]+p(histHost[i], ySize, xSize);
	}
}


__global__ void correctColorDevice(unsigned char* imageCharCorrectDevice, unsigned char* imageCharDevice, float* histCDFDevice, int cdfmin, int ySize, int xSize, int numChannel) {
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<ySize*xSize*numChannel) {
		int initVal=(int)imageCharDevice[index];
		float x=(255*histCDFDevice[initVal]-cdfmin)/(float)(1-cdfmin);
		float maxPart=x>0? x:0;
		float minPart=maxPart<255? maxPart:255;
		int correctVal=(int)minPart;
		imageCharCorrectDevice[index]=(unsigned char)correctVal;
	}
}


void correctColorHost(unsigned char* imageCharCorrectDevice, unsigned char* imageCharDevice, float* histCDFDevice, int cdfmin, int ySize, int xSize, int numChannel) {
	dim3 dimBlock(1024, 1, 1);
	dim3 dimGrid(ceil(xSize*ySize*numChannel/(float)1024), 1, 1);
	correctColorDevice<<<dimGrid, dimBlock>>>(imageCharCorrectDevice, imageCharDevice, histCDFDevice, cdfmin, ySize, xSize, numChannel);
	cudaDeviceSynchronize();
	cuCheck(__LINE__);
}


__global__ void outputImageDevice(float* imageOutFloatDevice, unsigned char* imageCharCorrectDevice, int ySize, int xSize, int numChannel) {
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<ySize*xSize*numChannel) {
		imageOutFloatDevice[index]=(float)(imageCharCorrectDevice[index]);
	}
}


void outputImageHost(float* imageOutFloatDevice, unsigned char* imageCharCorrectDevice, int ySize, int xSize, int numChannel) {
	dim3 dimBlock(1024, 1, 1);
	dim3 dimGrid(ceil(xSize*ySize*numChannel/(float)1024), 1, 1);
	outputImageDevice<<<dimGrid, dimBlock>>>(imageOutFloatDevice, imageCharCorrectDevice, ySize, xSize, numChannel);
	cudaDeviceSynchronize();
	cuCheck(__LINE__);
}


void mp6(float* imageFloatHost, float* imageOutFloatHost, int ySize, int xSize, int numChannel) {
	float* imageFloatDevice;
	unsigned char* imageCharHost;
	unsigned char* imageCharDevice;
	unsigned char* imageGrayHost;
	unsigned char* imageGrayDevice;

	int imageFloatSize=ySize*xSize*numChannel*sizeof(float);
	int imageCharSize=ySize*xSize*numChannel*sizeof(unsigned char);
	int imageGraySize=ySize*xSize*sizeof(unsigned char);

	imageCharHost=(unsigned char*)malloc(imageCharSize);
	imageGrayHost=(unsigned char*)malloc(imageGraySize);
	cudaMalloc((void **) &imageFloatDevice, imageFloatSize);
	cudaMalloc((void **) &imageCharDevice, imageCharSize);
	cudaMalloc((void **) &imageGrayDevice, imageGraySize);
	cudaMemcpy(imageFloatDevice, imageFloatHost, imageFloatSize, cudaMemcpyHostToDevice);
	cuCheck(__LINE__);

	f2ucHost(imageCharDevice, imageFloatDevice, ySize, xSize, numChannel);
	cudaMemcpy(imageCharHost, imageCharDevice, imageCharSize, cudaMemcpyDeviceToHost);
	cuCheck(__LINE__);

	charWritter(imageCharHost, ySize, xSize, numChannel);

	uc2grHost(imageGrayDevice, imageCharDevice, ySize, xSize, numChannel);
	cudaMemcpy(imageGrayHost, imageGrayDevice, imageGraySize, cudaMemcpyDeviceToHost);
	cuCheck(__LINE__);
	grayWritter(imageGrayHost, ySize, xSize);

	int histSize=HISTOGRAM_LENGTH*sizeof(int);
	int* histHost;
	int* histDevice;
	histHost=(int*)malloc(histSize);
	cudaMalloc((void **) &histDevice, histSize);
	for(int i=0; i<HISTOGRAM_LENGTH; ++i) {
		histHost[i]=0;
	}
	cudaMemcpy(histDevice, histHost, histSize, cudaMemcpyHostToDevice);
	cuCheck(__LINE__);
	getHistHost(histDevice, imageGrayDevice, ySize, xSize);
	cudaMemcpy(histHost, histDevice, histSize, cudaMemcpyDeviceToHost);
	cuCheck(__LINE__);

	float* histCDFHost;
	int histCDFSize=HISTOGRAM_LENGTH*sizeof(float);
	histCDFHost=(float*)malloc(histCDFSize);
	getCDF(histCDFHost, histHost, ySize, xSize);
	// int* histCDFDevice;
	// cudaMalloc((void **) &histCDFDevice, histSize);
	// getHistCDFHost(histCDFDevice, histDevice, ySize, xSize);
	// cudaMemcpy(histCDFHost, histCDFDevice, histSize, cudaMemcpyDeviceToHost);
	// cuCheck(__LINE__);
	//printf("%d\n", histCDFHost[HISTOGRAM_LENGTH-1]);
	// int sum=0;
	// for(int i=0; i<HISTOGRAM_LENGTH; i++) {
	// 	sum+=histHost[i];
	// }
	// printf("%d\n", sum);

	int cdfmin=histCDFHost[0];
	for(int i=0; i<HISTOGRAM_LENGTH; ++i) {
		cdfmin=cdfmin<histCDFHost[i]? cdfmin:histCDFHost[i];
	}

	float* histCDFDevice;
	cudaMalloc((void **) &histCDFDevice, histCDFSize);
	cudaMemcpy(histCDFDevice, histCDFHost, histCDFSize, cudaMemcpyHostToDevice);
	unsigned char* imageCharCorrectHost;
	unsigned char* imageCharCorrectDevice;
	imageCharCorrectHost=(unsigned char*)malloc(imageCharSize);
	cudaMalloc((void **) &imageCharCorrectDevice, imageCharSize);
	cuCheck(__LINE__);
	correctColorHost(imageCharCorrectDevice, imageCharDevice, histCDFDevice, cdfmin, ySize, xSize, numChannel);
	cudaMemcpy(imageCharCorrectHost, imageCharCorrectDevice, imageCharSize, cudaMemcpyDeviceToHost);
	cuCheck(__LINE__);
	// int* histClampHost;
	// histClampHost=(int*)malloc(histSize);
	/*int* histClampDevice;
	cudaMalloc((void **) &histClampDevice, histSize);
	cudaMemcpy(histClampDevice, histCDFHost, histSize, cudaMemcpyHostToDevice);
	clampHost(histClampDevice, histCDFHost[0]);
	cudaMemcpy(histClampHost, histClampDevice, histSize, cudaMemcpyDeviceToHost);
	cuCheck(__LINE__);
	for(int i=0; i<HISTOGRAM_LENGTH; i++) {
		printf("%d ", histClampHost[i]);
		if(histClampHost[i]<0||histClampHost[i]>255) {
			printf("wrong!\n");
		}
	}
	printf("At line: %d\n", __LINE__);*/
	// cdf2clamp(histCDFHost, histClampHost);
	// for(int i=0; i<HISTOGRAM_LENGTH; i++) {
	// 	printf("%d ", histClampHost[i]);
	// 	if(histClampHost[i]>255) {
	// 		printf("wrong!\n");
	// 	}
	// }
	// printf("At line: %d\n", __LINE__);
	float* imageOutFloatDevice;
	cudaMalloc((void **) &imageOutFloatDevice, imageFloatSize);
	outputImageHost(imageOutFloatDevice, imageCharCorrectDevice, ySize, xSize, numChannel);
	cuCheck(__LINE__);
	cudaMemcpy(imageOutFloatHost, imageOutFloatDevice, imageFloatSize, cudaMemcpyDeviceToHost);

	free(imageCharHost);
	cudaFree(imageCharDevice);
	free(imageGrayHost);
	cudaFree(imageGrayDevice);
	free(histHost);
	cudaFree(histDevice);
	free(histCDFHost);
	cudaFree(histCDFDevice);
	free(imageCharCorrectHost);
	cudaFree(imageCharCorrectDevice);
	return;
}


int main() {
	float* image=NULL;
	int* ySize=(int*)malloc(sizeof(int));
	int* xSize=(int*)malloc(sizeof(int));
	int numChannel=3;
	ppmReader(&image, ySize, xSize);
	// printf("%f %f %f\n", image[0], image[1], image[2]);
	// printf("%f %f %f\n", image[1166397], image[1166398], image[1166399]);
	// printf("%d, %d\n", *ySize, *xSize);
	float* imageOut=(float*)malloc(numChannel*(*ySize)*(*xSize)*sizeof(float));
	mp6(image, imageOut, *ySize, *xSize, numChannel);
	floatWritter(imageOut, *ySize, *xSize, numChannel);
	free(image);
	free(ySize);
	free(xSize);
	free(imageOut);
	return 0;
}
