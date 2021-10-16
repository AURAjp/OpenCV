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
    Mat srcImg = imread("./img/in.jpg", IMREAD_COLOR);
    // 読み込んだ画像のNULLチェック
    if (srcImg.empty()) {
        return -1;
    }

    // 入力画像のアスペクト比
    double aspectRatio = (double)srcImg.rows / srcImg.cols;
    // 出力画像の横幅
    int width = 800;
    // アスペクト比を保持した高さ
    int height = aspectRatio * width;

    // リサイズ用画像領域の確保
    Mat resizeImg(height, width, srcImg.type());
    // リサイズ
    resize(srcImg, resizeImg, resizeImg.size());

    // Canny法によるエッジ検出用画像領域の確保
    Mat cannyImg;
    // しきい値100&300、ApertureSize=3、画像勾配の精度向上=true
    Canny(resizeImg, cannyImg, 100., 300., 3, true);
    // エッジ検出画像の色を反転
    bitwise_not(cannyImg, cannyImg);

    //Bilateralフィルタ処理用画像領域確保
    Mat bilateralImg;
    bilateralFilter(resizeImg, bilateralImg, 15, 30., 30., BORDER_DEFAULT);

    // 論理積を計算
    Mat outImg;
    bitwise_and(bilateralImg, cannyImg, outImg);

    // 出力画像を表示
    imshow("Canny", cannyImg);
    imshow("Bilateral", bilateralImg);
    imshow("outImg", outImg);

    // 出力画像を保存
    imwrite("canny.png", cannyImg);
    imwrite("bilateral.png", bilateralImg);
    imwrite("out.png", outImg);

    waitKey(0);

    return 0;
}