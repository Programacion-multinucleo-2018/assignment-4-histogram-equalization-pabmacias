#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <stdio.h>
#include <iostream>

using namespace std;

void histogram(const cv::Mat& input, cv::Mat& output) {
  float h[256] = {0};
  float h_s[256] = {0};

  float size = input.rows*input.cols;
  float normalize = 255/size;

  for (int i=0; i<input.rows; i++) {
    for (int j=0; j<input.cols; j++) {
      h[(int)input.at<uchar>(i,j)]++;
    }
  }

  for (int i=0; i<256; i++) {
    for (int j=0; j<=i; j++) {
      h_s[i] += h[j];
    }
    h_s[i] = h_s[i]*(normalize);
  }

  for (int i=0; i<output.rows; i++) {
    for (int j=0; j<output.cols; j++) {
      output.at<uchar>(i,j) = h_s[(int)input.at<uchar>(i,j)];
    }
  }
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
  histogram(output, output_n);

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
