#include <opencv4/opencv2/opencv.hpp>
#include <DipFantasy.h>
#include <math.h>
#include <queue>
#include <vector>
using namespace std;
using namespace cv;
using namespace DIP_Fantasy;

//合规的直线需要符合的要求
bool LineRule(DF_IMG &mat, double theta, double radius)
{

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

    //满足输出条件的线：上截矩在图像范围内，并且直线得是满足条件的梯形：/ \ 通过k值和upper以及under讨论得出
    if ((uppper < mat.GetColSize() * 1.3 && uppper >= mat.GetColSize() * (-0.3)) && ((k > 0 && (uppper >= under) || k < 0 && (uppper <= under))))
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
            }
            else
            {
                *input.GetPoint(i, j) = 0;
            }
        }
    }
}

//通过HSI模型得到满足要求的黄色线
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
    return output;
}

int main(int argc, char const *argv[])
{

    if (argc != 2)
    {
        return -1;
    }
    Mat image = imread(argv[1]);
    // Mat image = imread("input/example.jpg");

    DF_Color_IMG input(image);

    input.DoConvolution(DF_Kernel(GaussianKernel, 5));

    DF_Color_IMG y_mask = GetYellowLine(input);

    //得到白色线
    DF_Color_IMG w_mask = input;
    DF_TYPE_INT white_max = GetWhite(input);
    white_max *= 0.9;
    DF_TYPE_INT rgb_w[3] = {white_max, white_max, white_max};
    int color_radius = 70;
    w_mask.DoColorSlicing(rgb_w, color_radius);

    w_mask.DoPlus(y_mask);


    DF_IMG mask = w_mask.ToGrey();

    //删除天空
    DeleteSky(mask);
    mask.DoDilation(DF_Kernel(BoxKernel, 5));

    DF_IMG grey = mask;

    //边界提取
    mask.DoConvolution(DF_Kernel(SobelKernelX, 3));
    //霍夫变换
    HoughTransition HT(mask, 10);

    int count = 0;
    for (int i = 0; i < HT.node_queue.size(); i++)
    {
        HoughNode now = HT.node_queue.top();

        //满足输出条件的线
        if (LineRule(grey, now.theta_average, now.radius_average))
        {
            //输出每个满足条件的线的极坐标
            // DrawLineToImage(grey, now.radius_average, now.theta_average);
            if (count >= 4)
            {
                break;
            }
            cout << now.theta_average << " " << now.radius_average << endl;
            DrawLineToImage(mask, now.radius_average, now.theta_average);
            count++;
        }

        HT.node_queue.pop();
    }
}