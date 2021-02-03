#ifndef __DIP_FANTASY__
#define __DIP_FANTASY__
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
//所有可能用到的openCV的操作
namespace OCV_Util
{
    template <typename DF_TYPE>
    static DF_TYPE *GetPoint(Mat &input, int x, int y)
    {
        int row_max = input.rows;
        int col_max = input.cols * input.channels();
        if (x < 0 || x >= row_max || y < 0 || y >= col_max)
        {
            return NULL;
        }
        DF_TYPE *p = input.ptr<DF_TYPE>(x);

        return p + y * input.channels();
    }

} // namespace OCV_Util

//写一个继承

namespace DIP_Fantasy
{
    /********************************************************矩阵Mat对象*****************************************************/
    //更新行列大小
    void DF_Mat::UpdateSize(int rows, int cols)
    {
        this->col_size = cols;
        this->row_size = rows;
    }

    Mat DF_Mat::GetMat()
    {
        return this->OCV_Mat;
    }
    int DF_Mat::GetColSize()
    {
        return this->col_size;
    }
    int DF_Mat::GetRowSize()
    {
        return this->row_size;
    }
    DF_Mat::~DF_Mat()
    {
        OCV_Mat.~Mat();
    }

    /********************************************************卷积核Kernel对象*****************************************************/
    DF_Kernel::~DF_Kernel()
    {
    }
    DF_Kernel::DF_Kernel(PREDEFINED_KERNEL kernel_type, int size)
    {
        if (size % 2 == 0)
        {
            printf("error! kernel size should be odd number!\n");
            return;
        }

        if (kernel_type == GaussianKernel)
        {
            Mat temp = Mat::zeros(size, size, CV_32F);
            temp.copyTo(this->OCV_Mat);
            int row_max = size;

            float *p = this->OCV_Mat.ptr<float>(size / 2);

            *(p + (size / 2)) = 1.0;

            Size k_size;
            k_size.height = size;
            k_size.width = size;
            GaussianBlur(this->OCV_Mat, this->OCV_Mat, k_size, 0.3 * ((size - 1) * 0.5 - 1) + 0.8);
        }
        else if (kernel_type == DonutsKernel)
        {
            int thickness = 1;
            Mat temp = Mat::zeros(size, size, CV_32F);
            temp.copyTo(this->OCV_Mat);

            // this->OCV_Mat = Mat::zeros(size, size, CV_32F);
            //top left
            for (int i = 0; i < thickness; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    *this->GetPoint(i, j) = 1.0;
                    *this->GetPoint(j, i) = 1.0;
                }
            }
            for (int i = size - thickness; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    *this->GetPoint(i, j) = 1.0;
                    *this->GetPoint(j, i) = 1.0;
                }
            }
        }

        else if (kernel_type == SobelKernelX)
        {
            // this->OCV_Mat = Mat::zeros(3, 3, CV_32F);
            Mat temp = Mat::zeros(3, 3, CV_32F);
            temp.copyTo(this->OCV_Mat);
            *OCV_Util::GetPoint<DF_TYPE_FLOAT>(this->OCV_Mat, 0, 0) = -1;
            *OCV_Util::GetPoint<DF_TYPE_FLOAT>(this->OCV_Mat, 0, 2) = 1;
            *OCV_Util::GetPoint<DF_TYPE_FLOAT>(this->OCV_Mat, 1, 0) = -2;
            *OCV_Util::GetPoint<DF_TYPE_FLOAT>(this->OCV_Mat, 1, 2) = 2;
            *OCV_Util::GetPoint<DF_TYPE_FLOAT>(this->OCV_Mat, 2, 0) = -1;
            *OCV_Util::GetPoint<DF_TYPE_FLOAT>(this->OCV_Mat, 2, 2) = 1;
        }
        else if (kernel_type == SobelKernelY)
        {
            // this->OCV_Mat = Mat::zeros(3, 3, CV_32F);
            Mat temp = Mat::zeros(3, 3, CV_32F);
            temp.copyTo(this->OCV_Mat);
            *OCV_Util::GetPoint<DF_TYPE_FLOAT>(this->OCV_Mat, 0, 0) = 1;
            *OCV_Util::GetPoint<DF_TYPE_FLOAT>(this->OCV_Mat, 0, 1) = 2;
            *OCV_Util::GetPoint<DF_TYPE_FLOAT>(this->OCV_Mat, 0, 2) = 1;
            *OCV_Util::GetPoint<DF_TYPE_FLOAT>(this->OCV_Mat, 2, 0) = -1;
            *OCV_Util::GetPoint<DF_TYPE_FLOAT>(this->OCV_Mat, 2, 1) = -2;
            *OCV_Util::GetPoint<DF_TYPE_FLOAT>(this->OCV_Mat, 2, 2) = -1;
        }

        else if (kernel_type == BoxKernel)
        {
            Mat temp = Mat::zeros(size, size, CV_32F);
            temp.copyTo(this->OCV_Mat);
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    *GetPoint(i, j) = 1.0;
                }
            }
        }

        UpdateSize(this->OCV_Mat.rows, this->OCV_Mat.cols);
    }
    void DF_Kernel::Show()
    {
        cv::Mat dst;
        namedWindow("Display Kernel", WINDOW_AUTOSIZE);
        cv::normalize(this->OCV_Mat, dst, 0, 1, cv::NORM_MINMAX);
        imshow("Display Kernel", this->OCV_Mat); //imshow似乎只能显示整数的
        cv::waitKey(0);
    }
    DF_TYPE_FLOAT *DF_Kernel::GetPoint(int rows, int cols)
    {
        int row_max = GetRowSize();
        int col_max = GetColSize();
        if (rows < 0 || rows >= row_max || cols < 0 || cols >= col_max)
        {
            return (DF_TYPE_FLOAT *)&this->null_pixel;
        }
        DF_TYPE_FLOAT *p = this->OCV_Mat.ptr<DF_TYPE_FLOAT>(rows);

        return p + cols * this->OCV_Mat.channels();
    }

    /********************************************************图像IMG对象*****************************************************/
    void DF_IMG::DoMultiply(DF_IMG &mask)
    {

        for (int i = 0; i < row_size; i++)
        {
            for (int j = 0; j < col_size; j++)
            {
                if (this->GetPoint(i, j) != 0)
                {
                    if (*mask.GetPoint(i, j) == 0)
                    {
                        *this->GetPoint(i, j) = 0;
                    }
                }
            }
        }
    }
    //拷贝函数
    DF_IMG &DF_IMG::operator=(DF_IMG other)
    {
        other.OCV_Mat.copyTo(this->OCV_Mat);
        UpdateSize(this->OCV_Mat.rows, this->OCV_Mat.cols);
        return *this;
    }
    int *DF_IMG::GetHisgram()
    {
        int *hist = (int *)calloc(256, sizeof(int));

        for (int i = 0; i < this->GetRowSize(); i++)
        {
            for (int j = 0; j < this->GetColSize(); j++)
            {
                int sum = 0;
                for (int c = 0; c < 3; c++)
                {
                    sum += *(this->GetPoint(i, j) + c);
                }
                hist[sum / 3]++;
            }
        }
        return hist;
    }
    void DF_IMG::DoHistEqualization()
    {
        for (int c = 0; c < 3; c++)
        {

            DF_TYPE_INT map[256];
            long long hist[256] = {0};
            for (int i = 0; i < this->GetRowSize(); i++)
            {
                for (int j = 0; j < this->GetColSize(); j++)
                {
                    hist[*(this->GetPoint(i, j) + c)]++;
                }
            }
            long long num = 0;
            for (int i = 0; i < 256; i++)
            {
                num += hist[i];
                double val = ((double)num / (double)(this->GetColSize() * this->GetRowSize())) * 255;
                map[i] = (DF_TYPE_INT)val;
            }
            for (int i = 0; i < this->GetRowSize(); i++)
            {
                for (int j = 0; j < this->GetColSize(); j++)
                {
                    *(this->GetPoint(i, j) + c) = map[*(this->GetPoint(i, j) + c)];
                }
            }
        }
    }
    //拷贝构造函数
    DF_IMG::DF_IMG(const DF_IMG &other)
    {
        other.OCV_Mat.copyTo(this->OCV_Mat);
        UpdateSize(this->OCV_Mat.rows, this->OCV_Mat.cols);
        this->null_pixel = 0;
    }

    //从OpenCV构造
    DF_IMG::DF_IMG(Mat &OpenCV_Mat)
    {
        OpenCV_Mat.copyTo(this->OCV_Mat);
        UpdateSize(this->OCV_Mat.rows, this->OCV_Mat.cols);
        this->null_pixel = 0;
    }
    //直接输入大小构造全0矩阵
    DF_IMG::DF_IMG(int rows, int cols)
    {
        this->OCV_Mat = Mat::zeros(rows, cols, CV_8UC1);
        UpdateSize(rows, cols);
        this->null_pixel = 0;
    }

    DF_IMG::~DF_IMG()
    {
    }
    void DF_IMG::DoPlus(DF_IMG &other)
    {
        for (int c = 0; c < this->OCV_Mat.channels(); c++)
        {

            for (int i = 0; i < row_size; i++)
            {
                for (int j = 0; j < col_size; j++)
                {

                    *(this->GetPoint(i, j) + c) += *other.GetPoint(i, j);
                }
            }
        }
    }
    void DF_IMG::ConvertTo_OpenCV_Mat(Mat &destination)
    {
        this->OCV_Mat.copyTo(destination);
    }
    void DF_IMG::Show()
    {
        namedWindow("Display Image", WINDOW_AUTOSIZE);
        imshow("Display Image", this->OCV_Mat); //imshow似乎只能显示整数的
        waitKey(0);
    }
    DF_TYPE_INT *DF_IMG::GetPoint(int row, int col)
    {
        int row_max = GetRowSize();
        int col_max = GetColSize();
        if (row < 0 || row >= row_max || col < 0 || col >= col_max)
        {
            return &this->null_pixel;
        }
        uchar *p = this->OCV_Mat.ptr<uchar>(row);

        return p + col * this->OCV_Mat.channels();
    }
    void DF_IMG::DoConvolution(DF_Kernel kernel)
    {
        int row = this->row_size;
        int col = this->col_size;
        int kernel_row = kernel.GetRowSize();
        int kernel_col = kernel.GetColSize();

        int l = (kernel_row / 2);
        int u = (kernel_col / 2);

        Mat *temp = new Mat;
        this->OCV_Mat.copyTo(*temp);
        for (int c = 0; c < this->OCV_Mat.channels(); c++)
        {
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    DF_TYPE_INT *now_point = OCV_Util::GetPoint<DF_TYPE_INT>(OCV_Mat, i, j) + c;
                    if (now_point != NULL)
                    {
                        *now_point = 0;
                        int ans = 0;
                        for (int i2 = -l; i2 < l + 1; i2++)
                        {
                            for (int j2 = -u; j2 < u + 1; j2++)
                            {
                                DF_TYPE_INT *p = OCV_Util::GetPoint<DF_TYPE_INT>(*temp, i + i2, j + j2) + c;
                                if (p != NULL)
                                {
                                    double t = (*p);
                                    ans += t * (*kernel.GetPoint(i2 + l, j2 + u));
                                }
                            }
                        }

                        if (ans >= 255 || ans <= -256)
                        {
                            ans = 255;
                        }
                        if (ans <= 0)
                        {
                            ans = -ans;
                        }

                        *now_point = (uchar)(ans);
                    }
                }
            }
        }
        delete temp;
    }
    void DF_IMG::DoErosion(DF_Kernel kernel, DF_TYPE_INT Threshold)
    {
    }
    void DF_IMG::DoDilation(DF_Kernel kernel, DF_TYPE_INT Threshold)
    {
        int kernel_row = kernel.GetRowSize();
        int kernel_col = kernel.GetColSize();
        int l = kernel_row / 2;
        int u = kernel_col / 2;
        Mat *temp = new Mat;
        this->OCV_Mat.copyTo(*temp);
        for (int c = 0; c < this->OCV_Mat.channels(); c++)
        {
            for (int i = 0; i < row_size; i++)
            {
                for (int j = 0; j < col_size; j++)
                {
                    for (int i2 = -l; i2 < l + 1; i2++)
                    {
                        for (int j2 = -u; j2 < u + 1; j2++)
                        {
                            DF_TYPE_INT *p = OCV_Util::GetPoint<DF_TYPE_INT>(*temp, i + i2, j + j2) + c;
                            if (p != NULL && *p > Threshold && *kernel.GetPoint(i2 + l, j2 + u) > 0)
                            {
                                *GetPoint(i, j) = 255;
                                double t = (*p);
                                goto next_pixel;
                            }
                        }
                    }
                next_pixel:;
                }
            }
        }
        delete temp;
    }

    /********************************************************彩图RGB_IMG对象*****************************************************/
    void DF_RGB_IMG::DoColorEnhancement(int map[256], RGB channel)
    {
        for (int i = 0; i < this->GetRowSize(); i++)
        {
            for (int j = 0; j < this->GetColSize(); j++)
            {
                *(this->GetPoint(i, j, channel)) = map[*(this->GetPoint(i, j, channel))];
            }
        }
    }
    void DF_RGB_IMG::DoPlus(DF_RGB_IMG &other)
    {
        for (int c = 0; c < 3; c++)
        {

            for (int i = 0; i < row_size; i++)
            {
                for (int j = 0; j < col_size; j++)
                {

                    *this->GetPoint(i, j, (RGB)c) += *other.GetPoint(i, j, (RGB)c);
                }
            }
        }
    }
    DF_RGB_IMG &DF_RGB_IMG::operator=(DF_RGB_IMG other)
    {
        other.OCV_Mat.copyTo(this->OCV_Mat);
        UpdateSize(this->OCV_Mat.rows, this->OCV_Mat.cols);
        return *this;
    }
    DF_IMG DF_RGB_IMG::ToGrey()
    {
        Mat grey;
        cvtColor(this->OCV_Mat, grey, COLOR_RGB2GRAY);
        return DF_IMG(grey);
    }
    DF_TYPE_INT *DF_RGB_IMG::GetPoint(int row, int col, RGB channel)
    {
        DF_TYPE_INT *rt = DF_IMG::GetPoint(row, col);
        if (rt != &this->null_pixel)
        {
            return DF_IMG::GetPoint(row, col) + channel;
        }
        return &this->null_pixel;
    }
    void DF_RGB_IMG::DoColorSlicing(DF_TYPE_INT RGB_Value[3], int radius)
    {
        for (int i = 0; i < this->row_size; i++)
        {
            for (int j = 0; j < this->col_size; j++)
            {
                int l_r = (int)*this->GetPoint(i, j, R) - (int)RGB_Value[0];
                int l_g = (int)*this->GetPoint(i, j, G) - (int)RGB_Value[1];
                int l_b = (int)*this->GetPoint(i, j, B) - (int)RGB_Value[2];
                if ((l_r * l_r + l_b * l_b + l_g * l_g > radius * radius))
                {
                    *this->GetPoint(i, j, R) = 0;
                    *this->GetPoint(i, j, G) = 0;
                    *this->GetPoint(i, j, B) = 0;
                }
            }
        }
    }
    void DF_RGB_IMG::DoMultiply(DF_IMG &mask)
    {
        for (int c = 0; c < 3; c++)
        {

            for (int i = 0; i < row_size; i++)
            {
                for (int j = 0; j < col_size; j++)
                {
                    if (this->GetPoint(i, j, (RGB)c) != 0)
                    {
                        if (*mask.GetPoint(i, j) == 0)
                        {
                            *this->GetPoint(i, j, (RGB)c) = 0;
                        }
                    }
                }
            }
        }
    }
    DF_RGB_IMG::DF_RGB_IMG(Mat &OpenCV_Mat) : DF_IMG(OpenCV_Mat)
    {
    }

    DF_RGB_IMG::~DF_RGB_IMG()
    {
    }

    DF_IMG ShiftIMG(DF_IMG &source, int row_up, int col_left)
    {
        DF_IMG rt_val(source.GetRowSize(), source.GetColSize());
        for (int c = 0; c < source.GetMat().channels(); c++)
        {
            for (int i = 0; i < source.GetRowSize(); i++)
            {
                for (int j = 0; j < source.GetColSize(); j++)
                {
                    *rt_val.GetPoint(i, j) = *source.GetPoint(i + row_up, j + col_left);
                }
            }
        }
        return rt_val;
    }
    DF_RGB_IMG ShiftIMG(DF_RGB_IMG &source, int row_up, int col_left)
    {
        DF_RGB_IMG rt_val = source;
        for (int c = 0; c < source.GetMat().channels(); c++)
        {
            for (int i = 0; i < source.GetRowSize(); i++)
            {
                for (int j = 0; j < source.GetColSize(); j++)
                {
                    *rt_val.GetPoint(i, j, (DF_RGB_IMG::RGB)c) = *source.GetPoint(i + row_up, j + col_left, (DF_RGB_IMG::RGB)c);
                }
            }
        }
        return rt_val;
    }

    void DrawLineToImage(DF_IMG &input, int radius, int theta)
    {
        int print_val = 255;
        int max_row = input.GetRowSize();
        double cost = cos(((double)theta / 180.0) * M_PI), sint = sin(((double)theta / 180.0) * M_PI);
        // double cost = cos(theta), sint = sin(theta);
        for (int i = 0; i < max_row; i++)
        {
            int y = (radius - i * cost) / sint;
            *input.GetPoint(i, y) = print_val;
        }
    }

    void HoughNode::InsertSon(int son_theta, int son_radius, int son_hough_value)
    {
        HoughSon *new_son = new HoughSon;
        this->hough_value_sum += son_hough_value;
        new_son->next_son = NULL;
        new_son->theta = son_theta;
        new_son->radius = son_radius;
        new_son->hough_value = son_hough_value;
        if (son_hough_value > this->son_max_hough_value)
        {
            this->son_max_hough_value = son_hough_value;
            this->theta_average = son_theta;
            this->theta_average = son_radius;
        }
        if (this->son == NULL)
        {
            this->son = new_son;
        }
        else
        {
            HoughSon *cursor = this->son;
            while (cursor->next_son != NULL)
            {
                cursor = cursor->next_son;
            }
            cursor->next_son = new_son;
        }
    }
    HoughNode::HoughNode(const HoughNode &other)
    {
        // other = *this;
        *this = other;
        this->son = NULL; //防止double free，直接把孩子节点给删除掉
    }
    HoughNode::HoughNode(int hough_value, int theta, int radius)
    {
        this->hough_value_sum = hough_value;
        this->this_hough_value = hough_value;
        this->son_max_hough_value = hough_value;

        this->theta = theta;
        this->radius = radius;
        this->theta_average = theta;
        this->radius_average = radius;
        this->hit_count = 1;
        this->next_node = NULL;
        this->son = NULL;
    }
    HoughNode::~HoughNode()
    {
        DeleteSon(this->son);
    }

    void HoughNode::DeleteSon(HoughSon *cursor)
    {
        if (cursor == NULL)
        {
            return;
        }
        DeleteSon(cursor->next_son);

        delete cursor;
    }

    bool operator<(HoughNode a, HoughNode b)
    {
        return a.hough_value_sum < b.hough_value_sum;
    }

    int *HoughTransition::GetHoughPoint(int theta, int radius)
    {
        if (theta > 180 || theta < -90 || radius < -diag_size || radius > diag_size)
        {
            printf("ouch!");
            return &this->null_point;
        }

        // return this->hough_mat + (90 + 180 + 1) * (radius + diag_size) + (theta + 90);
        return this->hough_mat + (theta + 90) * (2 * diag_size + 1) + radius + diag_size;
    }

    HoughTransition::HoughTransition(DF_IMG input, DF_TYPE_INT Threshold)
    {
        int radius_range = 100;
        int theta_range = 20;
        row_size = input.GetRowSize();
        col_size = input.GetColSize();
        max_rc_size = row_size;
        if (col_size > max_rc_size)
        {
            max_rc_size = col_size;
        }
        diag_size = sqrt(row_size * row_size + col_size * col_size) + 0.5;
        this->hough_mat = (int *)calloc((diag_size + 1) * 2 * (180 + 90 + 1), sizeof(int));

        long long val = 0;
        long long val_point = 0;
        for (int i = 0; i < row_size; i++)
        {
            for (int j = 0; j < col_size; j++)
            {
                if (*input.GetPoint(i, j) > Threshold)
                {
                    for (int theta = -90; theta <= 180; theta++)
                    {
                        int radius = i * cos(((double)theta / 180.0) * M_PI) + j * sin(((double)theta / 180.0) * M_PI);
                        int *point = GetHoughPoint(theta, radius);
                        if (*point == 0)
                        {
                            val_point++;
                        }
                        (*point) += 1;
                        val++;
                    }
                }
            }
        }
        long long hough_point_threshold = val / val_point * 2;
        for (int i = -diag_size; i <= diag_size; i++)
        {
            for (int j = -90; j <= 180; j++)
            {
                int point_value = *GetHoughPoint(j, i);
                if (point_value > hough_point_threshold)
                {
                    HoughNode *cursor = this->hough_HEAD;
                    while (true)
                    {
                        //首个节点
                        if (hough_HEAD == NULL)
                        {
                            HoughNode *new_node = new HoughNode(point_value, j, i);
                            hough_HEAD = new_node;
                            break;
                        }
                        //防止程序炸
                        if (cursor == NULL)
                        {
                            printf("ouch!\n");
                            break;
                        }
                        //放子节点
                        if ((cursor->radius_average - i <= radius_range && cursor->radius_average - i >= -radius_range) && (cursor->theta_average - j <= theta_range && cursor->theta_average - j >= -theta_range))
                        {
                            cursor->InsertSon(j, i, point_value);
                            break;
                        }
                        //建立新节点
                        else if (cursor->next_node == NULL)
                        {
                            HoughNode *new_node = new HoughNode(point_value, j, i);
                            cursor->next_node = new_node;
                            break;
                        }
                        cursor = cursor->next_node;
                    }
                }
            }
        }
        //求每个节点的平均

        HoughNode *cursor = this->hough_HEAD;
        while (true)
        {
            if (cursor == NULL)
            {
                break;
            }
            node_queue.push(*cursor);
            cursor = cursor->next_node;
        }
    }

    HoughTransition::~HoughTransition()
    {
        //删除节点简直坑死,还得递归
    }

} // namespace DIP_Fantasy

#endif