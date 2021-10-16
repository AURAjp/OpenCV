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

    // 画像をリサイズ
    Mat resizeImg = resizeImage(srcImg);

    // Canny法によるエッジ検出
    Mat cannyImg;

    Canny(resizeImg, cannyImg, 100, 400, 3, true);

    // 出力画像を表示
    imshow("cannyImg", cannyImg);
    // 出力画像を保存
    imwrite("./img/cannyImg.jpg", cannyImg);

    // キーボードが押されるまで処理を待つ
    waitKey(0);

    return 0;
}

Mat resizeImage(Mat image) {
    // 入力画像のアスペクト比
    double aspectRatio = (double)image.rows / image.cols;
    // 出力画像の横幅
    int width = 800;
    // アスペクト比を保持した高さ
    int height = aspectRatio * width;

    // リサイズ用画像領域の確保
    Mat resizeImg(height, width, image.type());
    // リサイズ
    resize(image, resizeImg, resizeImg.size());

    return resizeImg;
}