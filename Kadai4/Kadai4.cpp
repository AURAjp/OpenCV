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

void swap_image(Mat& in)
{
    /**
    * 以下に続く入れ替えのために、画像の両辺のサイズを偶数にしておく
    */
    in = in(Rect(0, 0, in.cols & -2, in.rows & -2));
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
    const int half_width = in.cols / 2;
    const int half_height = in.rows / 2;

    Mat tmp;

    Mat q0(in, Rect(0, 0, half_width, half_height));
    Mat q1(in, Rect(half_width, 0, half_width, half_height));
    Mat q2(in, Rect(0, half_height, half_width, half_height));
    Mat q3(in, Rect(half_width, half_height, half_width, half_height));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

// DFTの結果である複素画像を表示用の強度画像に変換する。
void convert_image_from_DFT(const Mat& in, Mat& out)
{
    /**
     * 複素画像の実部と虚部を2枚の画像に分離する。
     */
    Mat planes[2];
    split(in, planes);

    /**
     * 2次元ベクトルの大きさを求める
     */
    magnitude(planes[0], planes[1], out);

    /**
     * 表示用に対数に変換する。そのため各ピクセルに1を加算しておく。
     */
    out += Scalar::all(1);
    log(out, out);

    swap_image(out);

    // 濃淡を強調するために、画像最小値を0に、最大値を1にするように正規化する。
    normalize(out, out, 0, 1, CV_MINMAX);
}

// 逆DFTで得られた画像を実画像に変換する。
void convert_image_from_IDFT(const Mat& in, const::Mat& origin, Mat& out)
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

void convert_CV_32FC2(const Mat& in, Mat& out)
{
    // 実部のみのimageと虚部を0で初期化したMatの配列
    Mat RealIamginary[] = {
        Mat_<float>(in), Mat::zeros(in.size(), CV_32F)
    };
    // 配列を合成
    merge(RealIamginary, 2, out);
}

void my_resize(const Mat& in, Mat& out)
{
    // 入力画像のアスペクト比
    const double aspect_ratio = (double)in.cols / in.rows;
    // 出力画像の横幅
    constexpr int WIDTH = 800;
    // アスペクト比を保持した高さ
    const int height = round(WIDTH / aspect_ratio);
    // リサイズ用画像領域の確保
    Mat resize_img(height, WIDTH, in.type());
    // リサイズ
    resize(in, resize_img, resize_img.size());
    resize_img.copyTo(out);
}

// ローパスフィルタ
void low_pass_filter(Mat& in)
{
    for (int i = 0; i < in.rows; i++)
    {
        for (int j = 0; j < in.cols; j++)
        {
            if (i > 100 && j > 100)
            {
                in.at<Vec2f>(i, j)[0] = 0;
                in.at<Vec2f>(i, j)[1] = 0;
            }
        }
    }
}

int main()
{
    // 画像の読み込み
    Mat src_img = imread("./img/in.jpg", IMREAD_GRAYSCALE);
    // 読み込んだ画像のNULLチェック
    if (src_img.empty())
    {
        return -1;
    }

    // 800 * ? のサイズにアスペクト比を保持したままリサイズ
    Mat resize_img;
    my_resize(src_img, resize_img);

    // フーリエ変換用画像領域の確保
    Mat complex_image;
    convert_CV_32FC2(resize_img, complex_image);

    // フーリエ変換
    dft(complex_image, complex_image);
    // フーリエ変換の結果の可視化
    Mat power_spectrum_image;
    convert_image_from_DFT(complex_image, power_spectrum_image);

    // ローパスフィルタをかける用にクローンをとっておく
    Mat dft_image = complex_image.clone();

    // 逆フーリエ変換
    idft(complex_image, complex_image);
    // 逆フーリエ変換の結果の可視化
    Mat idft_image;
    convert_image_from_IDFT(complex_image, resize_img, idft_image);

    // ローパスフィルタ
    low_pass_filter(dft_image);
    // ローパスフィルタの結果の可視化
    Mat low_pass_power_spectrum_image;
    convert_image_from_DFT(dft_image, low_pass_power_spectrum_image);
    // ローパスフィルタをかけた画像を逆フーリエ変換
    idft(dft_image, dft_image);
    // ローパスフィルタをかけた画像の逆フーリエ変換の結果の可視化
    Mat low_pass_image;
    convert_image_from_IDFT(dft_image, resize_img, low_pass_image);

    // 結果表示
    imshow("original", resize_img);
    imshow("dft", power_spectrum_image);
    imshow("idft", idft_image);
    imshow("low_pass_power_spectrum_image", low_pass_power_spectrum_image);
    imshow("low_pass_image", low_pass_image);

    /**
     * 結果画像の保存.<br>
     * 表示画像は浮動小数表現で値域が[0,1]なので255を掛けたものを入力とする.
     */
    imwrite("./img/original.png", resize_img);
    imwrite("./img/dft.png", power_spectrum_image * 255);
    imwrite("./img/idft.png", idft_image * 255);
    imwrite("./img/low_pass_power_spectrum_image.png", low_pass_power_spectrum_image * 255);
    imwrite("./img/low_pass_image.png", low_pass_image * 255);

    waitKey(0);

    return 0;
}