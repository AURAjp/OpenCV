#include <opencv2/opencv.hpp>
#include <vector>

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

/**
 *  800 x ? のサイズへアスペクト比を保ったままリサイズする.
 *
 * @param in 入力画像
 * @param out 出力画像
 */
void my_resize(const Mat& in, Mat& out)
{
    //! 入力画像のアスペクト比
    const double aspect_ratio = (double)in.cols / in.rows;
    //! 出力画像の横幅
    constexpr int WIDTH = 800;
    //! アスペクト比を保持した高さ
    const int height = round(WIDTH / aspect_ratio);

    // リサイズ用画像領域の確保
    Mat resize_img(height, WIDTH, in.type());
    // リサイズ
    resize(in, resize_img, resize_img.size());
    resize_img.copyTo(out);
}

int main()
{
    //! 画像の読み込み
    const Mat src_img = imread("../img/in.jpg", IMREAD_COLOR);
    // 読み込んだ画像のNULLチェック
    if (src_img.empty()) { return -1; }

    // 読み込んだ画像のリサイズ
    Mat resize_img;
    my_resize(src_img, resize_img);

    // BGRからYCrCbへ変換
    Mat YCrCb_image;
    cvtColor(resize_img, YCrCb_image, CV_BGR2YCrCb);

    // YCrCbそれぞれのチャンネルに分解
    vector<Mat> planes;
    split(resize_img, planes);

    // 出力用にクローンをとっておく
    Mat Y  = planes[0].clone();
    Mat Cr = planes[1].clone();
    Mat Cb = planes[2].clone();

    // 結果表示
    imshow("ycrcb_image", YCrCb_image);
    imshow("y_image", Y);
    imshow("cr_image", Cr);
    imshow("cb_image", Cb);

    imwrite("../img/kadai6/YCbCr.png", YCrCb_image);
    imwrite("../img/kadai6/Y.png", Y);
    imwrite("../img/kadai6/Cr.png", Cr);
    imwrite("../img/kadai6/Cb.png", Cb);

    // キーボードが押されるまでwait
    waitKey(0);

    return 0;
}