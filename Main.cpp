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
    /*
    测试项目:
    + 构造，析构，拷贝函数
    + 卷积
    + 相乘
    + 相加
    + Sobel
    + 平移
    */

    // Mat image = imread("/home/rinka/Documents/DIP-Fantasy/input/DataSet/0531/1492626749527113213/20.jpg");
    // Mat image = imread("/home/rinka/Documents/DIP-Fantasy/input/DataSet/0531/1492626726476963805/20.jpg");
    Mat image = imread("/home/rinka/Documents/DIP-Fantasy/input/DataSet/0531/1492629968347062256/20.jpg");

    DF_RGB_IMG input(image);
    // input.Show();
    DF_IMG grey = input.ToGrey();
    DF_Kernel *gaussian_kernel = new DF_Kernel(GaussianKernel, 5);

    for (int i = 0; i < 2; i++)
    {
        grey.DoConvolution(*gaussian_kernel);
    }

    DF_IMG grey_y = grey;
    grey_y.DoConvolution(DF_Kernel(SobelKernelY, 3));
    grey_y.DoThreshold(100);

    DF_Kernel *sobel = new DF_Kernel(SobelKernelX, 3);
    grey.DoConvolution(*sobel);

    grey.DoThreshold(100);
    // grey.DoPlus(grey_y);
    //获得图像蒙版
    int color_radius = 100;
    DF_TYPE_INT rgb_y[3] = {0xFE, 0xD1, 0x86};
    DF_TYPE_INT rgb_w[3] = {0xE0, 0xE0, 0xE0};
    DF_RGB_IMG y_mask = input;
    DF_RGB_IMG w_mask = input;
    y_mask.DoColorSlicing(rgb_y, color_radius);
    w_mask.DoColorSlicing(rgb_w, color_radius);
    w_mask.DoPlus(y_mask);

    DF_IMG mask = w_mask.ToGrey();

    for (int i = 0; i < 160; i++)
    {
        for (int j = 0; j < mask.GetColSize(); j++)
        {
            *mask.GetPoint(i, j) = 0;
        }
    }
    for (int i = 160; i < mask.GetRowSize(); i++)
    {
        for (int j = 0; j < mask.GetColSize() - i*2 + 20; j++)
        {
            *mask.GetPoint(i, j) = 0;
        }
      
    }

    mask.DoDilation(DF_Kernel(BoxKernel, 5), 50);


    grey.DoMultiply(mask);

    grey.DoDilation(DF_Kernel(BoxKernel, 3), 50);
    grey.DoErosion(DF_Kernel(BoxKernel, 3), 50);

    grey.Show();
    HoughTransition HT(grey, 50);
    for (int i = 0; i < 10; i++)
    {
        //可以再设一个K和b的值判断法，有些k和b不可能是车道线
        HoughNode temp = HT.node_queue.top();
        DrawLineToImage(grey, temp.radius_average, temp.theta_average);
        HT.node_queue.pop();
    }
    grey.Show();
    /*
    Todo:
    - 设计一个梯形的遮罩
    - 找到合适的颜色增强方法（直方图均衡不行，可能需要使用HSI）
    - 设计一个弧度转y=kx+b的斜率、截矩限定筛选法
    */
}