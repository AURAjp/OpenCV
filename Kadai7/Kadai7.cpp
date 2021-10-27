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
    //! 画像をグレースケールで読み込み
    const Mat in_img  = imread("../img/kadai7/in.png" , IMREAD_GRAYSCALE);
    const Mat tmp_img = imread("../img/kadai7/tmp.png", IMREAD_GRAYSCALE);
    // 読み込んだ画像のNULLチェック
    if (in_img.empty() || tmp_img.empty()) { return -1; }

    //! 類似の箇所の出力結果を保存するための領域
    Mat SSD, CC, ZNCC;

    // 類似箇所を検索
    matchTemplate(in_img, tmp_img, SSD , CV_TM_SQDIFF       );
    matchTemplate(in_img, tmp_img, CC  , CV_TM_CCORR        );
    matchTemplate(in_img, tmp_img, ZNCC, CV_TM_CCOEFF_NORMED);

    //! 探索結果の位置
    Point pos_SSD;
    Point pos_CC;
    Point pos_ZNCC;

    // 相互相関マップの最大値検索
    minMaxLoc(SSD , NULL, NULL, NULL, &pos_SSD );
    minMaxLoc(CC  , NULL, NULL, NULL, &pos_CC  );
    minMaxLoc(ZNCC, NULL, NULL, NULL, &pos_ZNCC);

    /**
     * 類似箇所を色付きの四角で囲みたいため、
     * 出力用画像はグレースケールではなくBGRで用意する
     */
    Mat out_img = in_img.clone();
    cvtColor(out_img, out_img, CV_GRAY2BGR);

    // 類似箇所を色付きの四角で囲む
    rectangle(out_img, Rect(pos_SSD.x , pos_SSD.y , tmp_img.cols, tmp_img.rows), Scalar(0, 0, 255), 2);
    rectangle(out_img, Rect(pos_CC.x  , pos_CC.y  , tmp_img.cols, tmp_img.rows), Scalar(0, 255, 0), 2);
    rectangle(out_img, Rect(pos_ZNCC.x, pos_ZNCC.y, tmp_img.cols, tmp_img.rows), Scalar(255, 0, 0), 2);

    // 結果表示
    imshow("out", out_img);

    imwrite("../img/kadai7/out.png", out_img);

    // キーボードが押されるまでwait
    waitKey(0);

    return 0;
}