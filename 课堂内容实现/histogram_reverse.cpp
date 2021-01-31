#include <DipFantasy.h>
#include <opencv4/opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;
/*
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
*/

void My_Sobel(Mat &input)
{
    cvtColor(input, input, COLOR_RGB2GRAY);

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

    my_convolution(input, t);
    namedWindow("Display Image2", WINDOW_AUTOSIZE);
    imshow("Display Image2", t); //imshow似乎只能显示整数的
    // Mat draw;
    // t.convertTo(draw, CV_8U, 2, 0); //alpha：放大倍数，beta：放大倍数加上的偏移量，这个难道不会溢出?

    waitKey(0);
}

void My_Hist(Mat &input)
{
    uchar map[256];
    long long hist[256] = {0};
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            uchar *p = GetPoint(input, i, j);
            if (p != NULL)
            {
                hist[*p] += 1;
            }
        }
    }
    long long num = 0;
    for (int i = 0; i < 256; i++)
    {
        num += hist[i];
        double val = ((double)num / (double)(input.rows * input.cols)) * 255;
        map[i] = (uchar)val;
    }

    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            uchar *p = GetPoint(input, i, j);
            if (p != NULL)
            {
                *p = map[*p];
            }
        }
    }
    namedWindow("Display Image0", WINDOW_AUTOSIZE);
    imshow("Display Image0", input); //imshow似乎只能显示整数的
    imwrite("histogram_equalization.png", input);
    //逆变换映射
    int rev_map[256];
    for (int i = 0; i < 256; i++)
    {
        rev_map[i] = -1; //标记元素
    }

    for (int i = 0; i < 256; i++)
    {
        rev_map[map[i]] = i;
    }
    uchar now_val = 0;
    for (int i = 0; i < 256; i++)
    {
        if (rev_map[i] != -1)
        {
            now_val = rev_map[i];
        }
        else
        {
            rev_map[i] = now_val;
        }
    }
    //逆变换图像
    for (int i = 0; i < input.rows; i++)
    {
        for (int j = 0; j < input.cols; j++)
        {
            uchar *p = GetPoint(input, i, j);
            if (p != NULL)
            {
                *p = rev_map[*p];
            }
        }
    }
    imwrite("histogram_equalization_reversed.png", input);
}

int main(int argc, char const *argv[])
{
    if (argc < 2)
    {
        return -1;
    }
    Mat image;
    image = imread(argv[1], 1);
    cvtColor(image, image, COLOR_RGB2GRAY);

    namedWindow("Display Image1", WINDOW_AUTOSIZE);
    imshow("Display Image1", image); //imshow似乎只能显示整数的

    My_Hist(image);

    namedWindow("Display Image2", WINDOW_AUTOSIZE);
    imshow("Display Image2", image); //imshow似乎只能显示整数的

    waitKey(0);

    return 0;
}