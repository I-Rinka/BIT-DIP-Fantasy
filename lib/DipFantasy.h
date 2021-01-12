#ifndef __DIP_FANTASY__
#define __DIP_FANTASY__
#include <opencv4/opencv2/opencv.hpp>
using namespace cv;
uchar *GetPoint(Mat &input, int x, int y);

//到时候使用自己的对象和接口来表示
void my_convolution(Mat &input, Mat &output);
#endif