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
	// 読み込んだ画像のNULLチェック
	if (src_img.empty()) {
		return -1;
	}

	// 入力画像のアスペクト比
	double aspect_ratio = (double)src_img.rows / src_img.cols;
	// 出力画像の横幅
	constexpr int width = 800;
	// アスペクト比を保持した高さ
	int height = aspect_ratio * width;

	// リサイズ用画像領域の確保
	Mat resize_img(height, width, src_img.type());
	// リサイズ
	resize(src_img, resize_img, resize_img.size());

	// Canny法によるエッジ検出用画像領域の確保
	Mat canny_img;
	// しきい値100&300、ApertureSize=3、画像勾配の精度向上=true
	Canny(resize_img, canny_img, 100., 300., 3, true);
	// エッジ検出画像の色を反転
	bitwise_not(canny_img, canny_img);
	// 論理積を計算するためにCV_8UのC1からCV_8UのC3に変換
	cvtColor(canny_img, canny_img, CV_GRAY2BGR);

	// Bilateralフィルタ処理用画像領域確保
	Mat bilateral_img;
	// Bilateralフィルタ適用
	bilateralFilter(resize_img, bilateral_img, 15, 30., 30., BORDER_DEFAULT);

	// 出力画像領域の確保
	Mat out_img;
	// 論理積を計算
	bitwise_and(bilateral_img, canny_img, out_img);

	// 出力画像を表示
	imshow("Canny", canny_img);
	imshow("Bilateral", bilateral_img);
	imshow("Out", out_img);

	// 出力画像を保存
	imwrite("./img/canny.png", canny_img);
	imwrite("./img/bilateral.png", bilateral_img);
	imwrite("./img/out.png", out_img);

	// キーボードが押されるまで処理を待つ
	waitKey(0);

	return 0;
}