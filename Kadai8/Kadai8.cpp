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
 * ランダム画像を生成する. 
 */
void create_random_image(const Mat& in, Mat& out)
{
    // 入力画像をCV32FC1に変換しておく
    Mat template_img;
    in.convertTo(template_img, CV_32FC1);

    vector<Mat> random;

    Point2f center = Point2f(template_img.cols / 2.f, template_img.rows / 2.f);
    double scale = 1.;
    for (int i = 0; i < 15; i++)
    {
        double degree = 24. * i;
        Mat change_src = getRotationMatrix2D(center, degree, scale);
        Mat rotated_img;
        warpAffine(template_img, rotated_img, change_src, template_img.size());
        random.push_back(rotated_img);
    }

    Mat tmp;
    Mat tmp1 = random[0];
    Mat tmp2 = random[5];
    Mat tmp3 = random[10];

    for (int i = 1; i < 5; i++)
    {
        hconcat(tmp1, random[i], tmp1);
        hconcat(tmp2, random[i + 5], tmp2);
        hconcat(tmp3, random[i + 10], tmp3);
    }
    vconcat(tmp1, tmp3, tmp);
    vconcat(tmp, tmp2, tmp);

    // 出力画像はCV8UC1に戻しておく
    tmp.convertTo(tmp, CV_8UC1);
    out = tmp;
}

int main()
{
    //! 画像をグレースケールで読み込み
    const Mat src_img = imread("../img/kadai8/template.png", IMREAD_GRAYSCALE);
    // 読み込んだ画像のNULLチェック
    if (src_img.empty()) { return -1; }

    // ランダムイメージを生成する
    Mat random_img;
    create_random_image(src_img, random_img);

    // ランダムイメージのサイズを 800 * 450 にする
    const int pad_width  = (800 - random_img.cols) / 2;
    const int pad_height = (450 - random_img.rows) / 2;
    Mat input_img;
    copyMakeBorder(random_img, input_img,
        pad_height, pad_height, pad_width, pad_width,
        BORDER_CONSTANT, Scalar(0, 0, 0));

    // 結果表示
    imshow("tmp_img", tmp_img);

    // キーボードが押されるまでwait
    waitKey(0);

    return 0;
}
