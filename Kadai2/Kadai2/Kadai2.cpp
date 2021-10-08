#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

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

int main() {

    // 画像ファイルを読み込む
    Mat srcImg    = imread("./img/in.jpg");
    // フィルタ用画像をグレースケールで読み込む
    Mat filterImg = imread("./img/psf.png", 0);
    // 読み込んだ画像のNULLチェック
    if (srcImg.empty() || filterImg.empty()) {
        return -1;
    }

    // 入力画像のアスペクト比
    double aspectRatio = (double) srcImg.rows / srcImg.cols;
    // 出力画像の横幅
    int width = 800;
    // アスペクト比を保持した高さ
    int height = aspectRatio * width;

    // リサイズ用画像領域の確保
    Mat resizeImg(height, width, srcImg.type());
    // リサイズ
    resize(srcImg, resizeImg, resizeImg.size());

    // 画像をdouble型に変換
    filterImg.convertTo(filterImg, CV_64F);
    // 総和を求める
    double sumImg = (double) sum(filterImg)[0];
    // 総和で割る（総和が1 になる＝画像明るさ保存）
    filterImg = filterImg * (1. / sumImg);

    // 出力用画像領域の確保
    Mat outImg(height, width, srcImg.type());
    // リサイズした画像にフィルターをかける
    filter2D(resizeImg, outImg, 8, filterImg);

    // 出力画像を表示
    imshow("img", outImg);
    // 出力画像を保存
    imwrite("./img/out.jpg", outImg);

    // キーボードが押されるまで処理を待つ
    waitKey(0);

    return 0;
}