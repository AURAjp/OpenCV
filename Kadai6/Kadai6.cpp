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

    // CrとCbを 30 x 30 にリサイズ
    resize(planes[1], planes[1],
        Size(), 30. / planes[1].cols, 30. / planes[1].rows);
    resize(planes[2], planes[2],
        Size(), 30. / planes[2].cols, 30. / planes[2].rows);

    // 出力用にクローンをとっておく
    Mat Cr2 = planes[1].clone();
    Mat Cb2 = planes[2].clone();

    // CrとCbを元の大きさに再リサイズ
    resize(planes[1], planes[1],
        Size(), resize_img.cols / 30., resize_img.rows / 30.);
    resize(planes[2], planes[2],
        Size(), resize_img.cols / 30., resize_img.rows / 30.);

    // 出力用にクローンをとっておく
    Mat Cr3 = planes[1].clone();
    Mat Cb3 = planes[2].clone();

    Mat out;
    merge(planes, out);

    // 結果表示
    imshow("resize", resize_img);
    imshow("YCrCb", YCrCb_image);
    imshow("Y", Y);
    imshow("Cr", Cr);
    imshow("Cb", Cb);
    imshow("Cr2", Cr2);
    imshow("Cb2", Cb2);
    imshow("Cr3", Cr3);
    imshow("Cb3", Cb3);
    imshow("out", out);

    imwrite("../img/kadai6/resize_img.png", resize_img);
    imwrite("../img/kadai6/YCbCr.png", YCrCb_image);
    imwrite("../img/kadai6/Y.png", Y);
    imwrite("../img/kadai6/Cr.png", Cr);
    imwrite("../img/kadai6/Cb.png", Cb);
    imwrite("../img/kadai6/Cr2.png", Cr2);
    imwrite("../img/kadai6/Cb2.png", Cb2);
    imwrite("../img/kadai6/Cr3.png", Cr3);
    imwrite("../img/kadai6/Cb3.png", Cb3);
    imwrite("../img/kadai6/out.png", out);

    // キーボードが押されるまでwait
    waitKey(0);

    return 0;
}