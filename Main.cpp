#include <DipFantasy.h>
#include <opencv4/opencv2/opencv.hpp>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

using namespace cv;

//极坐标下的霍夫变换，霍夫变换中的一个点就对应直角坐标系中的一根线
class Hough_Fantasy
{
private:
    unsigned int *hough_mat_pointer;
    //排序?
    unsigned int *sorted_pointer;

    Mat input_mat;

    //最大的长度
    int max_p;

    //阈值
    int threshold;

    //在霍夫变换的图像上的点加权
    void DrawInHoughMat(int x, int y);

    unsigned int *GetHoughPoint(int theta, int p);

public:
    Hough_Fantasy(Mat &input, int threshold);
    ~Hough_Fantasy();

    //开始霍夫变换
    void DoHoughTransition();

    //得到在霍夫坐标系中当前值最大的那个点的极坐标参数
    void GetOnePoint(int &p_return, int &theta_return);

    //将极坐标的θ和半径转换为直角坐标的斜率和截距
    void ConvertPoleToXY(int p_input, int theta_input, int &k_return, int &b_return);

    //获得一条线的斜率和截距的参数
    void GetOneLine(int &k_return, int &b_return);
};
void Hough_Fantasy::GetOnePoint(int &p_return, int &theta_return)
{
    unsigned int max = 0;
    int rt_theta = -1, rt_p = -1;
    for (int theta = -90; theta <= 180; theta++)
    {
        for (int p = -this->max_p; p < this->max_p; p++)
        {
            unsigned int *point = GetHoughPoint(theta, p);
            if (point != NULL)
            {
                if (*point > max)
                {
                    max = *point;
                    rt_theta = theta;
                    rt_p = p;
                }
            }
        }
    }

    p_return = rt_p;
    theta_return = rt_theta;
}
unsigned int *Hough_Fantasy::GetHoughPoint(int theta, int p)
{
    //因为用的是偏移的，所以最后返回霍夫坐标的时候也要记得减去偏移量!
    //p有可能是送进来负数的。这里还可以再加一个边界值判断
    return this->hough_mat_pointer + ((90 + 180) * theta + p + max_p);
}

void Hough_Fantasy::DrawInHoughMat(int x, int y)
{
    //theta的范围
    for (int i = -90; i <= 180; i++)
    {
        //P=xcosθ+ysinθ
        int P = x * cos(i) + y * sin(i);
        unsigned int *point = GetHoughPoint(i, P);
        if (point != NULL)
        {
            *point = (*point) + 1;
        }
    }
    //以后性能优化可以在这里加个缓存，判断是不是一条线
}

void Hough_Fantasy::DoHoughTransition()
{
    int rows = this->input_mat.rows;
    int cols = this->input_mat.cols;
    this->max_p = (int)(sqrt(rows * rows + cols * cols) + 0.5);

    //长是θ，而宽是两倍的最大截距
    this->hough_mat_pointer = (unsigned int *)calloc(this->max_p * 2 * (90 + 180), sizeof(unsigned int));

    if (this->hough_mat_pointer == NULL)
    {
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < this->input_mat.rows; i++)
    {
        for (int j = 0; j < this->input_mat.cols; j++)
        {
            //遍历目标图像
            uchar *p = GetPoint(this->input_mat, i, j);
            if (p != NULL)
            {
                //如果p的值大于阈值，则送去做霍夫
                if (*p >= this->threshold)
                {
                    DrawInHoughMat(i, j);
                }
            }
        }
    }
}

//阈值，目前只支持输入灰度图像
Hough_Fantasy::Hough_Fantasy(Mat &input, int threshold)
{
    this->threshold = threshold;
    this->input_mat = input;
}

Hough_Fantasy::~Hough_Fantasy()
{
    free(this->hough_mat_pointer);
}

int main(int argc, char const *argv[])
{

    return 0;
}
