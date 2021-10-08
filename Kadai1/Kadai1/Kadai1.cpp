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
    Mat srcImg = imread("./img/in.jpg");
    if (srcImg.empty()) { 
        return -1;
    }

    // 入力画像のアスペクト比
    double aspectRatio = (double) srcImg.rows / srcImg.cols;
    // 出力画像の横幅
    int width = 800;
    // アスペクト比を保持した高さ
    int height = aspectRatio * width;

    // 出力用画像領域の確保
    Mat dstImg(height, width, srcImg.type());
    // リサイズ
    resize(srcImg, dstImg, dstImg.size());

    // 市松模様の数
    int checkerNum = 10;
    // 市松模様の幅
    double checkerWidth = width / checkerNum;
    // 市松模様の高さ
    double checkerHeight = height / checkerNum;

    // x,y軸についてそれぞれ0から画像の横幅,高さ分まで市松模様1個分ずつ増やしていく
    for (int y = 0; y < height; y += checkerHeight) {
        for (int x = 0; x < width; x += checkerWidth) {
            rectangle(dstImg, 
                Point(x, y), 
                Point(x + checkerWidth, y + checkerHeight),
                Scalar(0, 0, 255), -1, CV_AA);
            rectangle(dstImg,
                Point(x + checkerWidth, y + checkerHeight),
                Point(x + 2 * checkerWidth, y + 2 * checkerHeight),
                Scalar(0, 0, 255), -1, CV_AA);
        }
    }

    // 出力画像を表示
    imshow("img", dstImg);
    // 出力画像を保存
    imwrite("out.jpg", dstImg);

    // キーボードが押されるまで処理を待つ
    waitKey(0);

    return 0;
}