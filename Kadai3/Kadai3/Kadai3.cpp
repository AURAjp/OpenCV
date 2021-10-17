#include <opencv2/opencv.hpp>

using namespace cv;

#ifdef _DEBUG
// debug用のライブラリをリンク
#pragma comment(lib, "opencv_legacy2413d.lib")
#pragma comment(lib, "opencv_calib3d2413d.lib")
#pragma comment(lib, "opencv_core2413d.lib")
#pragma comment(lib, "opencv_objdetect2413d.lib")
#pragma comment(lib, "opencv_features2d2413d.lib")
#pragma comment(lib, "opencv_imgproc2413d.lib")
#pragma comment(lib, "opencv_highgui2413d.lib")
#else
// Release用のライブラリをリンク
#pragma comment(lib, "opencv_legacy2413.lib")
#pragma comment(lib, "opencv_core2413.lib")
#pragma comment(lib, "opencv_imgproc2413.lib")
#pragma comment(lib, "opencv_highgui2413.lib")
#pragma comment(lib, "opencv_objdetect2413.lib")
#pragma comment(lib, "opencv_features2d2413.lib")
#pragma comment(lib, "opencv_calib3d2413.lib")
#endif

void main() {
    Mat srcImg = imread("./img/in.jpg");
    resize(srcImg, srcImg, Size(800, 800 * srcImg.rows / srcImg.cols));

    Mat cannyImg;
    Canny(srcImg, cannyImg, 100, 300);
    bitwise_not(cannyImg, cannyImg);
    cvtColor(cannyImg, cannyImg, CV_GRAY2BGR);

    Mat bilateralImg;
    bilateralFilter(srcImg, bilateralImg, 15, 30, 30);

    Mat outImg;
    bitwise_and(bilateralImg, cannyImg, outImg);

    imwrite("./img/canny.png", cannyImg);
    imwrite("./img/bilateral.png", bilateralImg);
    imwrite("./img/out.png", outImg);

    waitKey(0);
    return;
}