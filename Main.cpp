#include <DipFantasy.h>
#include <opencv4/opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;
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