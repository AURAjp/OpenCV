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

int main()
{
    // 画像の読み込み
    const Mat src_img = imread("./img/in.jpg", IMREAD_GRAYSCALE);
    // 読み込んだ画像のNULLチェック
    if (src_img.empty())
    {
        return -1;
    }

    // 入力画像のアスペクト比
    const double aspect_ratio = (double)src_img.rows / src_img.cols;
    // 出力画像の横幅
    constexpr int WIDTH = 800;
    // アスペクト比を保持した高さ
    const int height = round(aspect_ratio * WIDTH);

    // リサイズ用画像領域の確保
    const Mat resize_img(height, WIDTH, src_img.type());
    // リサイズ
    resize(src_img, resize_img, resize_img.size());

    Mat out05_img;
    medianBlur(resize_img, out05_img,  5);
    Mat out11_img;
    medianBlur(resize_img, out11_img, 11);
    Mat out21_img;
    medianBlur(resize_img, out21_img, 21);

    // 結果表示
    imshow("out", out05_img);
    imshow("out", out11_img);
    imshow("out", out21_img);

    imwrite("./img/out5.png" , out05_img);
    imwrite("./img/out11.png", out11_img);
    imwrite("./img/out21.png", out21_img);

    // キーボードが押されるまでwait
    waitKey(0);

    return 0;
}