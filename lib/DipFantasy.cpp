#include <opencv4/opencv2/opencv.hpp>
#include <DipFantasy.h>
#define PX_TYPE uchar
using namespace cv;
uchar *GetPoint(Mat &input, int x, int y)
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

void my_convolution(Mat &input, Mat &output)
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

namespace DIP_Fantasy
{
    class DF_IMG
    {
    private:
        Mat OCV_Mat;
        PX_TYPE null_pixel;
        int row_size;
        int col_size;

        //改变行数和列数的记录
        void UpdateSize(int, int);

    public:
        DF_IMG(Mat &OpenCV_Mat);
        DF_IMG(int rows, int cols);
        ~DF_IMG();
        void ConvertTo_OpenCV_Mat(Mat &destination);

        void Show();

        Mat GetMat();

        //通过函数的方式减少对对象属性的可能的修改
        int GetRowSize();
        int GetColSize();

        //获得对应坐标的点
        PX_TYPE *GetPoint(int cols, int rows);
    };

    DF_IMG::DF_IMG(Mat &OpenCV_Mat)
    {
        OpenCV_Mat.copyTo(this->OCV_Mat);
        UpdateSize(this->OCV_Mat.rows, this->OCV_Mat.cols);
    }
    //create a img with
    DF_IMG::DF_IMG(int rows, int cols)
    {
        this->OCV_Mat = Mat::zeros(rows, cols, CV_8UC1);
        UpdateSize(rows, cols);
    }

    DF_IMG::~DF_IMG()
    {
        OCV_Mat.~Mat();
    }

    Mat DF_IMG::GetMat()
    {
        return this->OCV_Mat;
    }

    int DF_IMG::GetColSize()
    {
        return this->col_size;
    }
    int DF_IMG::GetRowSize()
    {
        return this->row_size;
    }

    //更新行列大小
    void DF_IMG::UpdateSize(int rows, int cols)
    {
        this->col_size = cols;
        this->row_size = rows;
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

} // namespace DIP_Fantasy

using namespace DIP_Fantasy;
int main(int argc, char const *argv[])
{
    Mat input = imread("/home/rinka/Documents/DIP-Fantasy/input/line.png");
    DF_IMG my_img(input);
    DF_IMG zeroIMG(200,200);
    zeroIMG.Show();
    return 0;
}
