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
    Mat src_img = imread("./img/in.jpg");
    if (src_img.empty()) {
        return -1;
    }

    // 入力画像のアスペクト比
    double aspect_ratio = (double)src_img.rows / src_img.cols;
    // 出力画像の横幅
    constexpr int width = 800;
    // アスペクト比を保持した高さ
    int height = aspect_ratio * width;

    // 出力用画像領域の確保
    Mat dst_img(height, width, src_img.type());
    // リサイズ
    resize(src_img, dst_img, dst_img.size());

    // 市松模様の数
    int checker_num = 10;
    // 市松模様の幅
	int checker_width = width / checker_num;
    // 市松模様の高さ
    int checker_height = height / checker_num;

    // x,y軸についてそれぞれ0から画像の横幅,高さ分まで市松模様1個分ずつ増やしていく
    for (int y = 0; y < height; y += checker_height * 2) {
		for (int x = 0; x < width; x += checker_width * 2) {
            rectangle(dstImg, 
                Point(x, y), 
                Point(x + checker_width, y + checker_height),
                Scalar(0, 255, 0), -1, CV_AA);
            rectangle(dstImg,
                Point(x + checker_width, y + checker_height),
                Point(x + checker_width * 2, y + checker_height * 2),
                Scalar(0, 255, 0), -1, CV_AA);
        }
    }

    // 出力画像を表示
    imshow("img", dst_img);
    // 出力画像を保存
    imwrite("./img/out.jpg", dst_img);

    // キーボードが押されるまで処理を待つ
    waitKey(0);

    return 0;
}