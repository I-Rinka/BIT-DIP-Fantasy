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
    Mat image = imread("/home/rinka/Documents/DIP-Fantasy/input/DataSet/0531/1492626718748019090/20.jpg");

    DF_Color_IMG input(image);

    input.Show();

    // DF_IMG grey=input.ToGrey();
    for (int i = 0; i < 3; i++)
    {
        input.DoConvolution(DF_Kernel(GaussianKernel, 5));
    }
    print(DF_Kernel(BoxKernel, 5).GetMat());

    input.Show();
    //获得图像蒙版
    int color_radius = 50;

    DF_TYPE_INT rgb_y[3] = {0xFE, 0xD1, 0x86};
    DF_TYPE_INT rgb_w[3] = {0xE0, 0xE0, 0xE0};
    DF_Color_IMG y_mask = input;
    DF_Color_IMG w_mask = input;
    y_mask.DoColorSlicing(rgb_y, color_radius);
    w_mask.DoColorSlicing(rgb_w, color_radius);
    w_mask.DoPlus(y_mask);

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
    cout << w_mask.GetColSize() << "row:" << w_mask.GetRowSize();
    // w_mask.Show();
    /*
    Todo:
    - 设计一个梯形的遮罩
    - 找到合适的颜色增强方法（直方图均衡不行，可能需要使用HSI）
    - 设计一个弧度转y=kx+b的斜率、截矩限定筛选法
    - HSI
    */
}