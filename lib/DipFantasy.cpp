#include <opencv4/opencv2/opencv.hpp>
#include <DipFantasy.h>
#define PX_TYPE uchar
using namespace cv;
//所有可能用到的openCV的操作
namespace OCV_Util
{
    static uchar *GetPoint(Mat &input, int x, int y)
    {
        int row_max = input.rows;
        int col_max = input.cols * input.channels();
        if (x < 0 || x >= row_max || y < 0 || y >= col_max)
        {
            return NULL;
        }
        uchar *p = input.ptr<uchar>(x);

        return p + y * input.channels();
    }

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
    }

} // namespace OCV_Util

//写一个继承

namespace DIP_Fantasy
{
    enum PREDEFINED_KERNEL
    {
        GaussianKernel,
        BoxKernel
    };
    class DF_Mat
    {
    private:
    protected:
        Mat OCV_Mat;
        PX_TYPE null_pixel;
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
        OCV_Mat.~Mat();
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
    }

    class DF_Kernel : public DF_Mat
    {
    private:
        /* data */
    public:
        DF_Kernel(PREDEFINED_KERNEL kenel_type, int size);
        ~DF_Kernel();
        double *GetPoint(int rows, int cols);
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
            print(this->OCV_Mat);

            Size k_size;
            k_size.height = size;
            k_size.width = size;
            GaussianBlur(this->OCV_Mat, this->OCV_Mat, k_size, 0.3 * ((size - 1) * 0.5 - 1) + 0.8);
            print(this->OCV_Mat);
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
    double *DF_Kernel::GetPoint(int rows, int cols)
    {
        int row_max = GetRowSize();
        int col_max = GetColSize();
        if (rows < 0 || rows >= row_max || cols < 0 || cols >= col_max)
        {
            return (double *)&this->null_pixel;
        }
        double *p = this->OCV_Mat.ptr<double>(rows);

        return p + cols * this->OCV_Mat.channels();
    }

    class DF_IMG : public DF_Mat
    {
    private:
        //改变行数和列数的记录

    public:
        DF_IMG(Mat &OpenCV_Mat);
        DF_IMG(int rows, int cols);
        //方便构造卷积核，注意size需要是奇数

        ~DF_IMG();
        void ConvertTo_OpenCV_Mat(Mat &destination);

        void Show();

        Mat GetMat();
        void DoConvolution(DF_Kernel kernel);

        //重载赋值运算符
        DF_IMG &operator=(DF_IMG other);
        //拷贝构造函数
        DF_IMG(const DF_IMG &other);

        //获得对应坐标的点
        PX_TYPE *GetPoint(int cols, int rows);
    };

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

    Mat DF_IMG::GetMat()
    {
        return this->OCV_Mat;
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
    PX_TYPE *DF_IMG::GetPoint(int rows, int cols)
    {
        int row_max = GetRowSize();
        int col_max = GetColSize();
        if (rows < 0 || rows >= row_max || cols < 0 || cols >= col_max)
        {
            return &this->null_pixel;
        }
        uchar *p = this->OCV_Mat.ptr<uchar>(rows);

        return p + cols * this->OCV_Mat.channels();
    }
    void DF_IMG::DoConvolution(DF_Kernel kernel)
    {
        int row = this->row_size;
        int col = this->col_size;
        int kernel_row = kernel.GetRowSize();
        int kernel_col = kernel.GetColSize();

        int l = -(kernel_row / 2);
        int u = -(kernel_col / 2);

        Mat *temp = new Mat;
        this->OCV_Mat.copyTo(*temp);
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                uchar *now_point = OCV_Util::GetPoint(OCV_Mat, i, j);
                if (now_point != NULL)
                {
                    *now_point = 0;
                    double ans = 0;
                    for (int i2 = l; i2 < -l + 1; i2++)
                    {
                        for (int j2 = u; j2 < -u + 1; j2++)
                        {
                            uchar *p = OCV_Util::GetPoint(*temp, i + i2, j + j2);
                            if (p != NULL)
                            {
                                double t = (*p);
                                ans += t * (*kernel.GetPoint(i2 + 1, j2 + 1));
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

        delete temp;
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
   */

    Mat input = imread("/home/rinka/Documents/DIP-Fantasy/input/class/2.bmp");
    cvtColor(input, input, COLOR_RGB2GRAY);
    printf("%d", input.channels());
    DF_IMG df_img(input);
    // df_img.Show();
    DF_Kernel g_kernel(GaussianKernel, 5);

    //未来可以优化一下高斯核，让它可以放大的可视化核心
    // g_kernel.Show();

    //高斯核是对的
    //现在卷积不知道为什么出来的全是黑的
    df_img.DoConvolution(g_kernel);
    df_img.Show();

    return 0;
}
