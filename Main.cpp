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

    //遮盖图像上部天空等地方
    for (int i = 0; i < 160; i++)
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
    DF_IMG grey = w_mask.ToGrey();
    grey.DoConvolution(DF_Kernel(SobelKernelX, 3));

    //霍夫变换
    HoughTransition HT(grey, 50);

    int count = 0;
    for (int i = 0; i < HT.node_queue.size(); i++)
    {
        HoughNode now = HT.node_queue.top();
        double cost = cos(((double)now.theta_average / 180.0) * M_PI), sint = sin(((double)now.theta_average / 180.0) * M_PI);

        //直线在图像顶部的截距
        double uppper = now.radius_average / sint;
        //直线在图像底部的截距
        double under = (now.radius_average - input.GetColSize() * cost) / sint;

        //直线的斜率
        double k = 0;
        if (sint != 0)
        {
            k = cost / sint;
        }

        //满足输出条件的线：上截矩在图像范围内，并且直线得是满足条件的梯形：/ \ 通过k值讨论
        if ((k > 0 && uppper < input.GetColSize() && uppper >= 0 && (uppper >= under) || k < 0 && uppper >= 0 && uppper <= input.GetColSize() && (uppper <= under))) //还要加个必须是梯形，上下大小也有关系，上面不能小于下面
        {
            //输出每个满足条件的线的极坐标
            DrawLineToImage(grey, now.radius_average, now.theta_average);
            count++;
            cout << now.theta_average << " " << now.radius_average << endl;
            if (count == 5)
            {
                break;
            }
        }

        HT.node_queue.pop();
    }
}