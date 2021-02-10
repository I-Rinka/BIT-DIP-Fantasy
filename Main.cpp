#include <opencv4/opencv2/opencv.hpp>
#include <DipFantasy.h>
#include <math.h>
#include <queue>
#include <vector>
using namespace std;
#define DF_TYPE_INT uchar
#define DF_TYPE_FLOAT float
using namespace cv;
using namespace DIP_Fantasy;

//合规的直线需要符合的要求
bool LineRule(DF_IMG &mat, double theta, double radius)
{
    // return true;
    double cost = cos((theta / 180.0) * M_PI), sint = sin((theta / 180.0) * M_PI);

    //直线在图像顶部的截距
    double uppper = radius / sint;
    //直线在图像底部的截距
    double under = radius - mat.GetColSize() * cost / sint;

    //直线的斜率
    double k = 0;
    if (sint != 0)
    {
        k = cost / sint;
    }

    //满足输出条件的线：上截矩在图像范围内，并且直线得是满足条件的梯形：/ \ 通过k值讨论
    //直线的规则
    if ((uppper < mat.GetColSize() * 1.3 && uppper >= mat.GetColSize() * (-0.3)))
    // && ((k > 0 && (uppper >= under) || k < 0 && (uppper <= under))))
    {
        return true;
    }
    else
    {
        return false;
    }
}

DF_TYPE_INT GetWhite(DF_Color_IMG &img)
{
    DF_TYPE_INT max = 0;
    for (int i = 400; i < img.GetRowSize(); i++)
    {
        for (int j = img.GetColSize() * 0.2; j < img.GetColSize() * 0.8; j++)
        {
            DF_TYPE_INT t_min = 255;
            for (int c = 0; c < 3; c++)
            {
                if (*img.GetPoint(i, j, (DF_Color_IMG::RGB)c) < t_min)
                {
                    t_min = *img.GetPoint(i, j, (DF_Color_IMG::RGB)c);
                }
            }
            if (max < t_min)
            {
                max = t_min;
            }
        }
    }
    return max;
}

void DeleteSky(DF_IMG &input)
{
    for (int i = 0; i < input.GetRowSize(); i++)
    {
        for (int j = 0; j < input.GetColSize(); j++)
        {
            int x = i - 480;
            int y = j - 640;
            if (i > 360 || (double)x * x / (200.0 * 200.0) + (double)(y * y) / (800.0 * 800.0) <= 1)
            {
                // *input.GetPoint(i, j) = 255;
            }
            else
            {
                *input.GetPoint(i, j) = 0;
            }
        }
    }
}

void DoSkel(DF_IMG &input)
{
    cv::Mat skel(input.GetMat().size(), CV_8UC1, cv::Scalar(0));
    cv::Mat temp;
    cv::Mat eroded;

    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

    bool done;
    do
    {
        cv::erode(input.GetMat(), eroded, element);
        cv::dilate(eroded, temp, element); // temp = open(img)
        cv::subtract(input.GetMat(), temp, temp);
        cv::bitwise_or(skel, temp, skel);
        eroded.copyTo(input.GetMat());

        done = (cv::countNonZero(input.GetMat()) == 0);
    } while (!done);

    skel.copyTo(input.GetMat());
}
DF_Color_IMG GetYellowLine(DF_Color_IMG &input)
{
    DF_Color_IMG output = input;
    for (int i = 0; i < output.GetRowSize(); i++)
    {
        for (int j = 0; j < output.GetColSize(); j++)
        {
            DF_TYPE_INT *R = output.GetPoint(i, j, DF_Color_IMG::R);
            DF_TYPE_INT *G = output.GetPoint(i, j, DF_Color_IMG::G);
            DF_TYPE_INT *B = output.GetPoint(i, j, DF_Color_IMG::B);
            int H = Get_HSI_H(*R, *G, *B);
            DF_TYPE_FLOAT S = Get_HSI_S(*R, *G, *B);
            int range = 5;
            if (S >= 0.40 && ((H >= 30 - range && H <= 30 + range) || (H >= 40 - range && H <= 40 + range)))
            {
            }
            else
            {
                *R = 0;
                *G = 0;
                *B = 0;
            }
        }
    }
    output.Show();
    return output;
}
int main(int argc, char const *argv[])
{

    // if (argc != 2)
    // {
    //     return -1;
    // }
    // Mat image = imread(argv[1]);
    // Mat image = imread("/home/rinka/Documents/DIP-Fantasy/input/example.jpg");
    // Mat image = imread("/home/rinka/Documents/DIP-Fantasy/input/example.jpg");
    // Mat image = imread("/home/rinka/Documents/DIP-Fantasy/input/DataSet/0601/1494453643540652356/20.jpg");
    Mat image = imread("input/DataSet/0531/1492626388446057821/20.jpg");
    // Mat image = imread("input/DataSet/0601/1494453731502184768/20.jpg");

    DF_Color_IMG input(image);

    input.DoConvolution(DF_Kernel(GaussianKernel, 5));

    DF_Color_IMG YL = GetYellowLine(input);
    // YL.Show();
}

//迷之点是怎么回事？