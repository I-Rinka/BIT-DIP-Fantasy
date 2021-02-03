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

    /*流程测试

    Mat image = imread("/home/rinka/Documents/DIP-Fantasy/input/DataSet/0601/1494453481611650216/20.jpg");
    DF_RGB_IMG input(image);

    input.Show();

    int color_radius = 100;
    DF_TYPE_INT rgb_y[3] = {0xFE, 0xD1, 0x86};
    DF_TYPE_INT rgb_w[3] = {0xE0, 0xE0, 0xE0};
    DF_RGB_IMG w_mask = input;
    DF_RGB_IMG y_mask = input;

    w_mask.DoColorSlicing(rgb_w, color_radius);
    y_mask.DoColorSlicing(rgb_y, color_radius);

    w_mask.DoPlus(y_mask);
    w_mask.Show();
    DF_IMG mask = w_mask.ToGrey();
    for (int i = 0; i < 160; i++)
    {
        for (int j = 0; j < mask.GetColSize(); j++)
        {
            *mask.GetPoint(i, j) = 0;
        }
    }
    for (int i = 0; i < 200; i++)
    {
        for (int j = 0; j < mask.GetColSize(); j++)
        {
            *mask.GetPoint(i, j) = 0;
        }
    }

    mask.Show();
    HoughTransition HT(mask, 50);
    for (int i = 0; i < 5; i++)
    {
        HoughNode line = HT.node_queue.top();
        DrawLineToImage(mask, line.radius_average, line.theta_average);

        HT.node_queue.pop();
    }

    mask.Show();*/
    DF_TYPE_INT rgb_y[3] = {0xFE, 0xD1, 0x86};
    DF_TYPE_INT rgb_w[3] = {0xE0, 0xE0, 0xE0};
    Mat image = imread("/home/rinka/Documents/DIP-Fantasy/input/example.jpg");
    
    DF_RGB_IMG input(image);
    input.Show();
    int map[256];
    //需要一个根据直方图判断的功能
    for (int i = 0; i < 256; i++)
    {
        map[i] = i;
        if (i < 100)
        {
            map[i] = i - 20;
        }
        if (i > 100)
        {
            map[i] = i + 20;
        }
        if (map[i] >= 256)
        {
            map[i] = 255;
        }
        if (map[i] < 0)
        {
            map[i] = 0;
        }
    }

    // for (int i = 0; i < 50; i++)
    // {
    //     map[i] = i / 2 + 1;
    // }
    // for (int i = 50; i < 256; i++)
    // {
    //     // map[i] = 61 + (i - 120) * 2;
    //     map[i] = i + 10;
    //     if (map[i] >= 256)
    //     {
    //         map[i] = 255;
    //     }
    // }
    for (int i = 0; i < 3; i++)
    {
        input.DoColorEnhancement(map, (DF_RGB_IMG::RGB)i);
    }
    input.Show();
}