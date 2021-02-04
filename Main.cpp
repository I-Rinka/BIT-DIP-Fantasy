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
    Mat image = imread("/home/rinka/Documents/DIP-Fantasy/input/DataSet/0531/1492626726476963805/20.jpg");
    // Mat image = imread("/home/rinka/Documents/DIP-Fantasy/input/DataSet/0531/1492626718748019090/20.jpg");
    // Mat image = imread("/home/rinka/Documents/DIP-Fantasy/input/line.png");

    DF_Color_IMG input(image);

    // DF_IMG grey=input.ToGrey();
    input.DoConvolution(DF_Kernel(GaussianKernel, 5));

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
    // w_mask.Show();
    /*
    Todo:
    - 设计一个梯形的遮罩
    - ~~找到合适的颜色增强方法（直方图均衡不行，可能需要使用HSI）~~
    - ~~设计一个弧度转y=kx+b的斜率、截矩限定筛选法🉑~~
    - ~~HSI~~
    - 转化成json的方法
    - 找到一个新的角度限定：比如k为负数的时候，必须在中线以内
    */
    DF_IMG grey = w_mask.ToGrey();
    grey.DoConvolution(DF_Kernel(SobelKernelX, 3));
    HoughTransition HT(grey, 50);

    cout << grey.GetColSize() << endl;
    cout << grey.GetColSize() << endl;
    int count = 0;
    for (int i = 0; i < HT.node_queue.size(); i++)
    {
        HoughNode now = HT.node_queue.top();
        // double cost = cos(((double)now.theta_average / 180.0) * M_PI);
        double cost = cos(((double)now.theta_average / 180.0) * M_PI), sint = sin(((double)now.theta_average / 180.0) * M_PI);
        double judge = now.radius_average / sint;
        //radius=sint*row+cost*col

        double k = 0;
        if (sint != 0)
        {
            k = cost / sint;
        }
        cout << judge << endl;
        //row=0时，线必须要图像内
        if ((judge >= 0 && judge <= input.GetColSize()) && (k < 0 && judge < input.GetColSize() / 2 || k > 0 && judge > input.GetColSize() / 2)) //还要加个必须是梯形，上下大小也有关系，上面不能小于下面
        {
            //这边的每个线就是输出的线
            DrawLineToImage(grey, now.radius_average, now.theta_average);
            count++;
            if (count == 5)
            {
                break;
            }
        }

        HT.node_queue.pop();
    }
    grey.Show();
}