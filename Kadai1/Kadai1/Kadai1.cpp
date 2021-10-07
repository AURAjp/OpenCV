#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#ifdef _DEBUG
// debug用のライブラリをリンク
#pragma comment(lib, "opencv_world453d.lib")
#else
// Release用のライブラリをリンク
#pragma comment(lib, "opencv_world453.lib")
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

    // 出力画像を表示
    imshow("img", dstImg);
    // 出力画像を保存
    imwrite("out.jpg", dstImg);

    // キーボードが押されるまで処理を待つ
    waitKey(0);

    return 0;
}