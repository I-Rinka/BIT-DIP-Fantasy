#include <opencv4/opencv2/opencv.hpp>
#include <stdio.h>
using namespace cv;

uchar *GetPoint(Mat &input, int x, int y)
{
    int row_max = input.rows;
    int col_max = input.cols * input.channels();
    if (x < 0 || x >= row_max || y < 0 || y >= col_max)
    {
        return NULL;
    }
    uchar *p = input.ptr<uchar>(x);

    return p + y * input.channels();
}

void my_convolution(Mat &input, Mat &output)
{
    int row = input.rows;
    int col = input.cols;

    // int kernel[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    int kernel[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

    input.copyTo(output);

    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            uchar *now_point = GetPoint(output, i, j);
            if (now_point != NULL)
            {
                *now_point = 0;
                int ans = 0;
                for (int i2 = -1; i2 < 2; i2++)
                {
                    for (int j2 = -1; j2 < 2; j2++)
                    {
                        uchar *p = GetPoint(input, i + i2, j + j2);
                        if (p != NULL)
                        {
                            int t = (*p);
                            ans += t * kernel[i2 + 1][j2 + 1];
                        }
                    }
                }

                if (ans >= 255 || ans <= -256)
                {
                    ans = 255;
                }
                if (ans <= 0)
                {
                    ans = -ans;
                }

                *now_point = (uchar)(ans);
            }
        }
    }
}
int main(int argc, char const *argv[])
{
    if (argc < 2)
    {
        return -1;
    }
    Mat image;
    image = imread(argv[1], 1);
    // Mat image(1000,1000,CV_8UC3,Scalar(255,200,255));

    cvtColor(image, image, COLOR_RGB2GRAY);

    Mat Blured;
    Size ksize;
    ksize.height = 9;
    ksize.width = 9;
    // GaussianBlur(image, Blured, ksize, 2.0, 0, BORDER_DEFAULT);

    // Mat Cannied;

    // Sobel(Blured, Cannied, CV_32F, 1, 0);
    Mat t;

    // Sobel(Blured, t, CV_8U, 0, 1);
    // namedWindow("Display Image", WINDOW_AUTOSIZE); //系统自己的sobel
    // imshow("Display Image", t);                    //imshow似乎只能显示整数的

    my_convolution(image, t);
    namedWindow("Display Image2", WINDOW_AUTOSIZE);
    imshow("Display Image2", t); //imshow似乎只能显示整数的
    // Mat draw;
    // t.convertTo(draw, CV_8U, 2, 0); //alpha：放大倍数，beta：放大倍数加上的偏移量，这个难道不会溢出?

    waitKey(0);

    return 0;
}
