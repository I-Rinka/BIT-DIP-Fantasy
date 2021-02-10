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
    Mat image = imread("/home/rinka/Documents/DIP-Fantasy/input/DataSet/0601/1494453785478375673/20.jpg");

    DF_Color_IMG input(image);

    //蒙板

    // input.Show();

    //图像平滑
    input.DoConvolution(DF_Kernel(GaussianKernel, 5));

    //color slicing获得黄色和白色的图像蒙版,即黄色和白色的车道线
    int color_radius = 50;
    DF_TYPE_INT white = GetWhite(input);
    // rgb(235, 167, 92)
    // #EFA15A
    // 0xD1;
    // DF_TYPE_INT rgb_y[3] = {0xFE, 0xD1, 0x86};
    DF_TYPE_INT rgb_y[3] = {220, 180, 100};
    // DF_TYPE_INT rgb_w[3] = {0xE0, 0xE0, 0xE0};
    DF_TYPE_INT rgb_w[3] = {white, white, white};
    DF_Color_IMG y_mask = input;
    DF_Color_IMG w_mask = input;

    y_mask.DoColorSlicing(rgb_y, color_radius);
    w_mask.DoColorSlicing(rgb_w, color_radius);
    w_mask.DoPlus(y_mask);
    // w_mask.Show();
    //边界提取

    // DF_IMG grey = input.ToGrey();
    // input.Show();

    DF_IMG mask = w_mask.ToGrey();
    // medianBlur(mask.GetMat(), mask.GetMat(), 5);
    DeleteSky(mask);
    mask.Show();
    //和蒙板相乘
    DF_IMG grey = input.ToGrey();
    grey.DoConvolution(DF_Kernel(SobelKernelX, 3));

    grey.DoMultiply(mask);
    // DoSkel(grey);
    grey.Show();

    // grey.DoPlus(w_mask);

    // grey.Show();

    // grey2.Show();
    // grey.Show();

    //霍夫变换

    HoughTransition HT(grey, 10);
    // grey.Show();
    int count = 0;
    for (int i = 0; i < HT.node_queue.size(); i++)
    {
        HoughNode now = HT.node_queue.top();
        //合规才输出
        if (LineRule(grey, now.theta_average, now.radius_average))
        {
            //输出4条直线效果最佳
            if (count >= 4)
            {
                break;
            }
            count++;
            cout << now.theta_average << " " << now.radius_average << endl;
            DrawLineToImage(grey, now.radius_average, now.theta_average);
            grey.Show();
        }
        HT.node_queue.pop();
    }
    grey.Show();
}

//迷之点是怎么回事？