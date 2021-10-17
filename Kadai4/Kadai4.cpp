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

// DFTの結果である複素画像を表示用の強度画像に変換する。
void create_fourier_magnitude_image_from_complex(const Mat& in, Mat& out)
{
	/**
	 * 複素画像の実部と虚部を2枚の画像に分離する。
	 */
	Mat planes[2];
	split(in, planes);

	/**
	 * magnitude()で二枚の画像から強度を算出する。
	 *
	 * 強度画像の各ピクセル値の擬似コード
	 * dst(x,y) = sqrt(pow( src1(x,y),2) + pow(src2(x,y),2) )
	 */
	magnitude(planes[0], planes[1], out);

	/**
	 * 表示用に対数に変換する。そのため各ピクセルに1を加算しておく。
	 */
	out += Scalar::all(1);
	log(out, out);

	/**
	 * 以下に続く入れ替えのために、画像の両辺のサイズを偶数にしておく
	 */
	out = out(Rect(0, 0, out.cols & -2, out.rows & -2));


	/**
	 * dft()で得られる画像を一般的な二次元DFTの表示様式にする。
	 *
	 * 具体的には以下のように画像の各四半分を入れ替える。
	 *
	 * +----+----+    +----+----+
	 * |    |    |    |    |    |
	 * | q0 | q1 |    | q3 | q2 |
	 * |    |    |    |    |    |
	 * +----+----+ -> +----+----+
	 * |    |    |    |    |    |
	 * | q2 | q3 |    | q1 | q0 |
	 * |    |    |    |    |    |
	 * +----+----+    +----+----+
	 */
	const int half_width = out.cols / 2;
	const int half_height = out.rows / 2;

	Mat tmp;

	Mat q0(out,
		Rect(0, 0, half_width, half_height));
	Mat q1(out,
		Rect(half_width, 0, half_width, half_height));
	Mat q2(out,
		Rect(0, half_height, half_width, half_height));
	Mat q3(out,
		Rect(half_width, half_height, half_width, half_height));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	// 濃淡を強調するために、画像最小値を0に、最大値を1にするように正規化する。
	normalize(out, out, 0, 1, CV_MINMAX);
}

// 逆DFTで得られた画像を実画像に変換する。
void create_inverse_fourier_image_from_complex(
	const Mat& in, const::Mat& origin, Mat& out)
{
	// 複素画像の実部と虚部を2枚の画像に分離する。
	Mat splitted_image[2];
	split(in, splitted_image);

	/**
	 * 実部について正規化を行う。
	 * 入力画像のサイズはDFT用に拡大されているので、原画像の同サイズにROIを設定して
	 * 縮小する。
	 */
	splitted_image[0](Rect(0, 0, origin.cols, origin.rows)).copyTo(out);
	normalize(out, out, 0, 1, CV_MINMAX);
}

/**
 * @fn
 * アスペクト比を保持したリサイズを行う.
 * @param image 画像データ
 **/
Mat resize(const Mat& image)
{
	// 入力画像のアスペクト比
	auto aspect_ratio = (double)image.rows / image.cols;
	// 出力画像の横幅
	constexpr auto width = 800;
	// アスペクト比を保持した高さ
	int height = aspect_ratio * width;

	// リサイズ用画像領域の確保
	Mat resize_img(height, width, image.type());
	// リサイズ
	resize(image, resize_img, resize_img.size());

	return resize_img;
}

int main()
{
	// 画像の読み込み
	auto src_img = imread("./img/in.jpg", IMREAD_GRAYSCALE);
	// 読み込んだ画像のNULLチェック
	if (src_img.empty())
	{
		return -1;
	}

	// 画像サイズをリサイズ
	auto resize_img = resize(src_img);

	//実部のみのimageと虚部を0で初期化したMatをplanes配列に入れる
	Mat planes[] = {
		Mat_<float>(resize_img), Mat::zeros(resize_img.size(), CV_32F)
	};
	// フーリエ変換のために適切なサイズの複素画像を作る
	Mat complex_image;
	//配列を合成
	merge(planes, 2, complex_image);

	// フーリエ変換
	dft(complex_image, complex_image);
	// フーリエ変換の結果の可視化
	Mat magnitude_image;
	create_fourier_magnitude_image_from_complex(complex_image, magnitude_image);

	// 逆フーリエ変換
	idft(complex_image, complex_image);
	// 逆フーリエ変換の結果の可視化
	Mat idft_image;
	create_inverse_fourier_image_from_complex(complex_image, resize_img, idft_image);

	// 結果表示
	imshow("original", resize_img);
	imshow("dft", magnitude_image);
	imshow("idft", idft_image);

	/**
	 * 結果画像の保存.<br>
	 * 表示画像は浮動小数表現で値域が[0,1]なので255を掛けたものを入力とする.
	 */
	imwrite("./img/dft.png", magnitude_image * 255);
	imwrite("./img/idft.png", idft_image * 255);

	waitKey(0);

	return 0;
}