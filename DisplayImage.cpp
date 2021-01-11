#include <opencv4/opencv2/opencv.hpp>
using namespace cv;
int main(int argc, char const *argv[])
{
    if (argc < 2)
    {
        return -1;
    }
    // Mat image;
    // image = imread(argv[1], 1);
    Mat image(1000,1000,CV_8UC3,Scalar(255,200,255));

    namedWindow("Dispay Image", WINDOW_AUTOSIZE);
    imshow("Display Image", image);
    waitKey(0);
    
    randu(image,Scalar::all(0),Scalar::all(255));


    if (!image.data)
    {
        return -1;
    }
    
    namedWindow("Dispay Image2", WINDOW_AUTOSIZE);

    imshow("Display Image", image);

    waitKey(0);

    return 0;
}
