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

//合规的直线
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

    //满足输出条件的线：上截矩在图像范围内，并且直线得是满足条件的梯形：/ \ 通过k值讨论
    //直线的规则
    if ((k > 0 && uppper < mat.GetColSize() && uppper >= 0 && (uppper >= under) || k < 0 && uppper >= 0 && uppper <= mat.GetColSize() && (uppper <= under)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

int main(int argc, char const *argv[])
{

    if (argc != 2)
    {
        return -1;
    }
    Mat image = imread(argv[1]);

    DF_Color_IMG input(image);

    input.DoConvolution(DF_Kernel(GaussianKernel, 5));

    //color slicing获得黄色和白色的图像蒙版
    int color_radius = 50;
    DF_TYPE_INT rgb_y[3] = {0xFE, 0xD1, 0x86};
    DF_TYPE_INT rgb_w[3] = {0xE0, 0xE0, 0xE0};
    DF_Color_IMG y_mask = input;
    DF_Color_IMG w_mask = input;

    y_mask.DoColorSlicing(rgb_y, color_radius);
    w_mask.DoColorSlicing(rgb_w, color_radius);
    w_mask.DoPlus(y_mask);

    input.DoConvolution(DF_Kernel(SobelKernelX, 3));
    //遮盖图像上部天空等地方
    for (int i = 0; i < 200; i++)
    {
        for (int c = 0; c < 3; c++)
        {

            for (int j = 0; j < w_mask.GetColSize(); j++)
            {
                *w_mask.GetPoint(i, j, (DF_Color_IMG::RGB)c) = 0;
            }
        }
    }

    //边界提取
    DF_IMG grey = input.ToGrey();

    grey.DoPlus(w_mask);
    // grey.DoConvolution(DF_Kernel(SobelKernelX, 3));

    //霍夫变换
    HoughTransition HT(grey, 50);

    int count = 0;
    vector<HoughNode> final_output;
    for (int i = 0; i < HT.node_queue.size(); i++)
    {
        HoughNode now = HT.node_queue.top();
        //合规才输出
        if (LineRule(grey, now.theta_average, now.radius_average))
        {
            count++;
            int judge = true;
            //第二重优化
            for (int j = 0; j < final_output.size(); j++)
            {
                double cost = cos(((double)now.theta_average / 180.0) * M_PI), sint = sin(((double)now.theta_average / 180.0) * M_PI);
                double under = (now.radius_average - 600 * cost) / sint;
                double cost2 = cos(((double)final_output[j].theta_average / 180.0) * M_PI), sint2 = sin(((double)final_output[j].theta_average / 180.0) * M_PI);
                double under2 = (final_output[j].radius_average - 600 * cost) / sint;
                if (abs(under - under2) > 500)
                {
                }
                else
                {
                    judge = false;
                }
            }
            if (judge)
            {
                final_output.push_back(now);
            }
            if (count > 5)
            {
                break;
            }
        }
        HT.node_queue.pop();
    }
    for (int i = 0; i < final_output.size(); i++)
    {
        HoughNode now = final_output[i];
        cout << now.theta_average << " " << now.radius_average << endl;
        // DrawLineToImage(output, now.radius_average, now.theta_average);
    }

    // output.Show();
}