#include <opencv4/opencv2/opencv.hpp>
#include <DipFantasy.h>
#define DF_TYPE_INT uchar
#define DF_TYPE_FLOAT float
using namespace cv;
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
    /*
    static void my_convolution(Mat &input, Mat &output)
    {
        int row = input.rows;
        int col = input.cols;

        // int kernel[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
        int kernel[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

        input.copyTo(output);

        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                uchar *now_point = GetPoint(output, i, j);
                if (now_point != NULL)
                {
                    *now_point = 0;
                    int ans = 0;
                    for (int i2 = -1; i2 < 2; i2++)
                    {
                        for (int j2 = -1; j2 < 2; j2++)
                        {
                            uchar *p = GetPoint(input, i + i2, j + j2);
                            if (p != NULL)
                            {
                                int t = (*p);
                                ans += t * kernel[i2 + 1][j2 + 1];
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
    }*/

} // namespace OCV_Util

//写一个继承

namespace DIP_Fantasy
{
    enum PREDEFINED_KERNEL
    {
        GaussianKernel,
        BoxKernel,
        SobelKernelX,
        SobelKernelY
    };
    class DF_Mat
    {
    private:
    protected:
        Mat OCV_Mat;
        DF_TYPE_INT null_pixel;
        int row_size;
        int col_size;
        void UpdateSize(int, int);

    public:
        DF_Mat(/* args */);
        ~DF_Mat();
        int GetColSize();
        int GetRowSize();
        Mat GetMat();
        void Show();
    };
    //更新行列大小
    void DF_Mat::UpdateSize(int rows, int cols)
    {
        this->col_size = cols;
        this->row_size = rows;
    }
    DF_Mat::DF_Mat(/* args */)
    {
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

    class DF_Kernel : public DF_Mat
    {
    private:
        /* data */
    public:
        DF_Kernel(PREDEFINED_KERNEL kenel_type, int size);
        ~DF_Kernel();
        DF_TYPE_FLOAT *GetPoint(int rows, int cols);
        void Show();
    };

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
            this->OCV_Mat = Mat::zeros(size, size, CV_32F);
            int row_max = size;

            float *p = this->OCV_Mat.ptr<float>(size / 2);

            *(p + (size / 2)) = 1.0;

            Size k_size;
            k_size.height = size;
            k_size.width = size;
            GaussianBlur(this->OCV_Mat, this->OCV_Mat, k_size, 0.3 * ((size - 1) * 0.5 - 1) + 0.8);
        }
        else if (kernel_type == SobelKernelX)
        {
            this->OCV_Mat = Mat::zeros(3, 3, CV_32F);
            *OCV_Util::GetPoint<DF_TYPE_FLOAT>(this->OCV_Mat, 0, 0) = -1;
            *OCV_Util::GetPoint<DF_TYPE_FLOAT>(this->OCV_Mat, 0, 2) = 1;
            *OCV_Util::GetPoint<DF_TYPE_FLOAT>(this->OCV_Mat, 1, 0) = -2;
            *OCV_Util::GetPoint<DF_TYPE_FLOAT>(this->OCV_Mat, 1, 2) = 2;
            *OCV_Util::GetPoint<DF_TYPE_FLOAT>(this->OCV_Mat, 2, 0) = -1;
            *OCV_Util::GetPoint<DF_TYPE_FLOAT>(this->OCV_Mat, 2, 2) = 1;
        }
        else if (kernel_type == SobelKernelY)
        {
            this->OCV_Mat = Mat::zeros(3, 3, CV_32F);
            *OCV_Util::GetPoint<DF_TYPE_FLOAT>(this->OCV_Mat, 0, 0) = 1;
            *OCV_Util::GetPoint<DF_TYPE_FLOAT>(this->OCV_Mat, 0, 1) = 2;
            *OCV_Util::GetPoint<DF_TYPE_FLOAT>(this->OCV_Mat, 0, 2) = 1;
            *OCV_Util::GetPoint<DF_TYPE_FLOAT>(this->OCV_Mat, 2, 0) = -1;
            *OCV_Util::GetPoint<DF_TYPE_FLOAT>(this->OCV_Mat, 2, 1) = -2;
            *OCV_Util::GetPoint<DF_TYPE_FLOAT>(this->OCV_Mat, 2, 2) = -1;
        }

        else if (kernel_type == BoxKernel)
        {
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

    class DF_IMG : public DF_Mat
    {
    private:
        //改变行数和列数的记录

    public:
        DF_IMG(Mat &OpenCV_Mat);
        //生成空白图，注意size需要是奇数
        DF_IMG(int rows, int cols);
        //拷贝构造函数
        DF_IMG(const DF_IMG &other);

        ~DF_IMG();
        void ConvertTo_OpenCV_Mat(Mat &destination);

        void Show();

        void DoConvolution(DF_Kernel kernel);
        void DoErosion(DF_Kernel kernel, DF_TYPE_INT Threshold);
        void DoDilation(DF_Kernel kernel, DF_TYPE_INT Threshold);
        void DoMultiply(DF_IMG &mask);
        void DoPlus(DF_IMG &other);
        //重载赋值运算符
        DF_IMG &operator=(DF_IMG other);
        //二维数组怎么重载

        //获得对应坐标的点
        DF_TYPE_INT *GetPoint(int cols, int rows);
    };
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
        printf("%d\n", this->OCV_Mat.channels());
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

    class DF_RGB_IMG : public DF_IMG
    {
    private:
        /* data */
    public:
        enum RGB
        {
            B,
            G,
            R
        };
        DF_RGB_IMG(Mat &OpenCV_Mat);
        //生成空白图，注意size需要是奇数
        //

        ~DF_RGB_IMG();
        //图像相乘

        DF_IMG ToGrey();
        DF_RGB_IMG &operator=(DF_RGB_IMG other);
        void DoMultiply(DF_IMG &mask);
        void DoPlus(DF_IMG &other);
        void DoPlus(DF_RGB_IMG &other);
        void DoColorSlicing(DF_TYPE_INT RGB_Value[3], int radius);
        DF_TYPE_INT *GetPoint(int row, int col, RGB channel);
    };
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

} // namespace DIP_Fantasy

using namespace DIP_Fantasy;
int main(int argc, char const *argv[])
{
    /*构造、析沟函数、拷贝函数测试
    Mat input = imread("/home/rinka/Documents/DIP-Fantasy/input/line.png");
    DF_IMG my_img(input);
    DF_IMG zeroIMG(200, 200);
    // zeroIMG.Show();
    zeroIMG = my_img;
    // zeroIMG.Show();
    DF_IMG *test = new DF_IMG(my_img);
    // test->Show();
    DF_IMG test2(*test);
    delete test;
    test2.Show();
    */

    /*卷积测试
    Mat input = imread("/home/rinka/Documents/DIP-Fantasy/input/class/2.bmp");
    cvtColor(input, input, COLOR_RGB2GRAY);
    printf("%d", input.channels());
    DF_IMG df_img(input);
    df_img.Show();
    DF_Kernel g_kernel(GaussianKernel, 5);

    //未来可以优化一下高斯核，让它可以放大的可视化核心
    // g_kernel.Show();

    //高斯核是对的
    //现在卷积不知道为什么出来的全是黑的
    df_img.DoConvolution(g_kernel);
    df_img.Show();
    //卷积是对的
   */

    /*彩图测试、相乘测试
    // Mat input = imread("/home/rinka/Documents/DIP-Fantasy/input/DataSet/0531/1492626726476963805/20.jpg");
    Mat input = imread("/home/rinka/Documents/DIP-Fantasy/input/DataSet/0531/1492626718748019090/20.jpg");
    DF_RGB_IMG rgb(input);
    // rgb.Show();

    DF_IMG grey = rgb.ToGrey();
    // grey.Show();

    // rgb.DoMultiply(grey);
    // rgb.Show();


    //图像相加
    DF_RGB_IMG white_mask = rgb;
    DF_RGB_IMG yellow_mask = rgb;
    DF_TYPE_INT w_rgb[3] = {0xF0, 0xF0, 0xF0};
    DF_TYPE_INT y_rgb[3] = {0xF5, (0xAC + 0xD4) / 2, (0x6B + 0x74) / 2};
    white_mask.DoColorSlicing(w_rgb, 50);
    yellow_mask.DoColorSlicing(y_rgb, 50);

    rgb.DoMultiply(yellow_mask);
    rgb.DoPlus(white_mask);
    rgb.Show();

    for (int i = 0; i < rgb.GetRowSize(); i++)
    {
        for (int j = 0; j < rgb.GetColSize(); j++)
        {
            // *rgb.GetPoint(i, j, DF_RGB_IMG::R) = 0;
            *rgb.GetPoint(i, j, DF_RGB_IMG::G) = 0;
            *rgb.GetPoint(i, j, DF_RGB_IMG::B) = 0;
        }
    }
    */

    /*sobel测试*/

    /*
    // rgb_mask_shifted.Show();
    cvtColor(input, input, COLOR_RGB2GRAY);
    DF_IMG gray(input);

    DF_IMG temp = gray;

    DF_Kernel sobelx(SobelKernelX, 3);

    //平移测试
    DF_IMG shifted = ShiftIMG(gray, 0, 10);
    gray.Show();
    shifted.Show();
*/

    /*
    gray.Show();
    // gray.DoConvolution(sobelx);
    temp.DoConvolution(DF_Kernel(SobelKernelX, 3));
    gray.DoConvolution(DF_Kernel(SobelKernelY, 3));
    gray.Show();
    gray.DoPlus(temp);
    gray.Show();

    temp.DoConvolution(DF_Kernel(SobelKernelX, 3));
    DF_TYPE_INT y_rgb[3] = {0xF5, (0xAC + 0xD4) / 2, (0x6B + 0x74) / 2};
    rgb_mask.DoColorSlicing(y_rgb, 50);
    temp.DoMultiply(rgb_mask);

    temp.Show();

    Mat shit = gray.GetMat();
    Sobel(shit, gray.GetMat(), -1, 1, 0, 3);
    gray.Show();
    return 0;
    */
    //新Sobel

    //OpenCV的Sobel:丢弃右边直线
}
