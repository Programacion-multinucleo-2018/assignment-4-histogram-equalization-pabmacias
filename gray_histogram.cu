#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include "common.h"

using namespace std;

__global__ void normalize_image(unsigned char* input, unsigned char* output, int width, int height, int grayWidthStep, float* h, float* h_s) {
  const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

  const int gray_tid = yIndex * grayWidthStep + xIndex;

  if (xIndex < width && yIndex < height) {
    output[gray_tid] = h_s[input[gray_tid]];
  }
}

__global__ void normalize_histogram(unsigned char* input, unsigned char* output, int width, int height, int grayWidthStep, float* h, float* h_s) {
  unsigned int nxy = threadIdx.x + threadIdx.y * blockDim.x;

  float size = width*height;
  float normalize = 255/size;

  if (nxy < 256 && blockIdx.x == 0 && blockIdx.y == 0){
    for (int i=0; i<=nxy; i++){
      h_s[nxy] += h[i];
    }
    h_s[nxy] = h_s[nxy]*normalize;
  }

}
__global__ void histogram(unsigned char* input, unsigned char* output, int width, int height, int grayWidthStep, float* h, float* h_s) {
  const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int nxy = threadIdx.x + threadIdx.y * blockDim.x;

  const int gray_tid = yIndex * grayWidthStep + xIndex;

  __shared__ int h_temp[256];

   if (nxy < 256) {
     h_temp[nxy] = 0;
   }

   __syncthreads();

   if (xIndex < width && yIndex < height) {
     atomicAdd(&h_temp[input[gray_tid]], 1);
   }

   __syncthreads();

   if (nxy < 256) {
     atomicAdd(&h[nxy], h_temp[nxy]);
   }

   __syncthreads();
}

void normalize(const cv::Mat& input, cv::Mat& output) {
  cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;
	// Calculate total number of bytes of input and output image
	// Step = cols * number of colors
	size_t grayBytes = output.step * output.rows;

	unsigned char *d_input, *d_output;
  float * h = {};
  float * h_s = {};

	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, grayBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, grayBytes), "CUDA Malloc Failed");
  SAFE_CALL(cudaMalloc(&h, 256*sizeof(float)), "CUDA Malloc Failed");
  SAFE_CALL(cudaMalloc(&h_s, 256*sizeof(float)), "CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
  SAFE_CALL(cudaMemcpy(d_input, input.ptr(), grayBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
  SAFE_CALL(cudaMemcpy(d_output, output.ptr(), grayBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	// Specify a reasonable block size
	const dim3 block(16,16);

	// Calculate grid size to cover the whole image
	// const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);
	const dim3 grid((input.cols)/block.x, (input.rows)/block.y);
	// printf("bgr_to_gray_kernel<<<(%d, %d) , (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);

	// Launch the color conversion kernel
  histogram<<<grid,block>>>(d_input, d_output, input.cols, input.rows, static_cast<int>(input.step), h, h_s);
  normalize_histogram<<<grid,block>>>(d_input, d_output, input.cols, input.rows, static_cast<int>(input.step), h, h_s);
  normalize_image<<<grid,block>>>(d_input, d_output, input.cols, input.rows, static_cast<int>(input.step), h, h_s);

	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, grayBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
}

int main(int argc, char *argv[]) {
  string imagePath;

	if(argc < 2)
		imagePath = "Images/dog1.jpeg";
  	else
  		imagePath = argv[1];

  cout << imagePath << endl;
	// Read input image from the disk
	cv::Mat input = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

	if (input.empty())
	{
		cout << "Image Not Found!" << std::endl;
		cin.get();
		return -1;
	}

	//Create output image
  // cv::Mat input_bw(input.rows, input.cols, CV_8UC1);
	cv::Mat output(input.rows, input.cols, CV_8UC1);

	cv::cvtColor(input, output, cv::COLOR_BGR2GRAY);

  cv::Mat output_n(input.rows, input.cols, CV_8UC1);
  normalize(output, output_n);

  //Allow the windows to resize
  namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);

	// output = input_bw.clone();
  imshow("Input", output);
	imshow("Output", output_n);

  //Wait for key press
	cv::waitKey();

	return 0;
}
