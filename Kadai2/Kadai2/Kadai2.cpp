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
	Mat src_img = imread("./img/in.jpg", IMREAD_COLOR);
	// フィルタ用画像をグレースケールで読み込む
	Mat filter_img = imread("./img/psf.png", IMREAD_GRAYSCALE);
	// 読み込んだ画像のNULLチェック
	if (src_img.empty() || filter_img.empty()) {
		return -1;
	}

	// 画像をdouble型に変換
	filter_img.convertTo(filter_img, CV_64F);
	// 総和を求める
	double sum_img = (double)sum(filter_img)[0];
	// 総和で割る（総和が1 になる＝画像明るさ保存）
	filter_img = filter_img * (1. / sum_img);

	// 出力用画像領域の確保
	Mat out_img(src_img.cols, src_img.rows, src_img.type());
	// 入力画像にフィルターをかける
	filter2D(src_img, out_img, 8, filter_img);

	// 出力画像を表示
	imshow("img", out_img);
	// 出力画像を保存
	imwrite("./img/out.jpg", out_img);

	// キーボードが押されるまで処理を待つ
	waitKey(0);

	return 0;
}