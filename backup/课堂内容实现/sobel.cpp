#include <opencv4/opencv2/opencv.hpp>
using namespace cv;
int main(int argc, char const *argv[])
{
    if (argc < 2)
    {
        return -1;
    }
    Mat image;
    image = imread(argv[1], 1);

    cvtColor(image, image, COLOR_RGB2GRAY);

    Mat Sobex;
    Sobel(image, Sobex, CV_32F, 1, 0); //sobel需要一张新的图像
    double MinVal, MaxVal;
    minMaxLoc(Sobex, &MinVal, &MaxVal);

    Mat draw;

    Sobex.convertTo(draw, CV_8U, 255.0 / (MaxVal - MinVal), 255); //alpha：放大倍数，beta：放大倍数加上的偏移量，这个难道不会溢出?

    if (!image.data)
    {
        return -1;
    }

    namedWindow("Dispay Image", WINDOW_AUTOSIZE);
    imshow("Display Image", draw);//imshow似乎只能显示整数的

    waitKey(0);

    return 0;
}
