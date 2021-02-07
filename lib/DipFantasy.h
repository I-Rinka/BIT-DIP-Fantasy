#include <opencv4/opencv2/opencv.hpp>
#include <math.h>
#include <queue>
#include <vector>
using namespace std;
#define DF_TYPE_INT uchar
#define DF_TYPE_FLOAT float
using namespace cv;

//所有可能用到的openCV的操作
namespace OCV_Util
{
    template <typename DF_TYPE>
    static DF_TYPE *GetPoint(Mat &input, int x, int y);

} // namespace OCV_Util

//写一个继承

namespace DIP_Fantasy
{
    enum PREDEFINED_KERNEL
    {
        GaussianKernel,
        BoxKernel,
        SobelKernelX,
        SobelKernelY,
        DonutsKernel,
        LineKernelY
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
        ~DF_Mat();
        int GetColSize();
        int GetRowSize();
        Mat GetMat();
        void Show();
    };

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
        void DoErosion(DF_Kernel kernel);
        void DoDilation(DF_Kernel kernel);
        void DoMultiply(DF_IMG &mask);
        void DoPlus(DF_IMG &other);
        void DoThreshold(DF_TYPE_INT Threshold);
        void DoHistEqualization();
        int *GetHisgram();
        //重载赋值运算符
        DF_IMG &operator=(DF_IMG other);
        //二维数组怎么重载

        //获得对应坐标的点
        DF_TYPE_INT *GetPoint(int cols, int rows);
    };

    class DF_Color_IMG : public DF_IMG
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
        enum HSI
        {
            H,
            S,
            I
        };
        DF_Color_IMG(Mat &OpenCV_Mat);
        //生成空白图，注意size需要是奇数
        //

        ~DF_Color_IMG();
        //图像相乘

        DF_IMG ToGrey();
        DF_Color_IMG &operator=(DF_Color_IMG other);
        void DoMultiply(DF_IMG &mask);
        void DoPlus(DF_IMG &other);
        void DoPlus(DF_Color_IMG &other);
        void DoColorEnhancement(int map[256], RGB channel);
        void DoThreshold(DF_TYPE_INT Threshold, RGB channel);
        void ConvertToHSI();
        void ConvertToRGB();
        void DoColorSlicing(DF_TYPE_INT *H_channel_value, int H_channel_num, int radius);
        DF_TYPE_INT *GetPoint(int row, int col, RGB channel);
        void DoColorSlicing(DF_TYPE_INT RGB_Value[3], int radius);
    };

    void DoShiftIMG(DF_IMG &source, int row_up, int col_left);
    DF_Color_IMG ShiftIMG(DF_Color_IMG &source, int row_up, int col_left);
    void DrawLineToImage(DF_IMG &input, int radius, int theta);
    struct HoughSon
    {
        int theta = 0;
        int radius = 0;
        int hough_value = 0;
        HoughSon *next_son;
    };
    struct HoughNode
    {
    public:
        int this_hough_value = 0;
        long long hough_value_sum = 0;
        int theta = 0;
        int radius = 0;
        int hit_count = 1;
        double theta_average = 0;
        double radius_average = 0;
        HoughNode *next_node;
        HoughSon *son;
        void InsertSon(int son_theta, int son_radius, int son_hough_value);
        HoughNode(const HoughNode &other);
        HoughNode(int hough_value, int theta, int radius);
        ~HoughNode();

    private:
        int son_max_hough_value;
        void DeleteSon(HoughSon *cursor);
    };

    bool operator<(HoughNode a, HoughNode b);
    class HoughTransition
    {
    private:
        int *hough_mat;
        int max_rc_size;
        int diag_size;
        int row_size;
        int col_size;
        int null_point;
        HoughNode *hough_HEAD = NULL;
        /* data */
        int *GetHoughPoint(int theta, int radius);

    public:
        priority_queue<HoughNode> node_queue;
        HoughTransition(DF_IMG input, DF_TYPE_INT Threshold);
        ~HoughTransition();
        int line_number = 0;
    };
    DF_TYPE_FLOAT Get_HSI_H(DF_TYPE_INT R, DF_TYPE_INT G, DF_TYPE_INT B);
    DF_TYPE_FLOAT Get_HSI_S(DF_TYPE_INT R, DF_TYPE_INT G, DF_TYPE_INT B);
    DF_TYPE_INT Get_HSI_I(DF_TYPE_INT R, DF_TYPE_INT G, DF_TYPE_INT B);
} // namespace DIP_Fantasy
