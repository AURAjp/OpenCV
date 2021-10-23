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

/**
* 自作のメディアンフィルタを行う関数
* 
* @param in 入力画像
* @param out 出力画像
* @param filter_size フィルタサイズ
*/
void my_medianBlur(const Mat& in, Mat& out, int filter_size)
{
    // フィルタサイズが1以下ならば入力画像と同じ画像を返す
    if (filter_size <= 1)
    {
        in.copyTo(out);
        return;
    }
    // フィルタサイズが奇数かどうかチェック
    CV_Assert(filter_size % 2 == 1);
}

int main()
{
    // 画像の読み込み
    const Mat src_img = imread("./img/in.jpg", IMREAD_GRAYSCALE);
    // 読み込んだ画像のNULLチェック
    if (src_img.empty())
    {
        return -1;
    }

    // 入力画像のアスペクト比
    const double aspect_ratio = (double)src_img.cols / src_img.rows;
    // 出力画像の横幅
    constexpr int WIDTH = 800;
    // アスペクト比を保持した高さ
    const int height = round(WIDTH / aspect_ratio);

    // リサイズ用画像領域の確保
    Mat resize_img(height, WIDTH, src_img.type());
    // リサイズ
    resize(src_img, resize_img, resize_img.size());

    // 境界処理用画像領域の確保
    constexpr int EXPIXEL = 100;
    Mat border_extend_img;
    copyMakeBorder(resize_img, border_extend_img, EXPIXEL, EXPIXEL, EXPIXEL, EXPIXEL, BORDER_REFLECT_101);

    // Median Blur
    Mat extend_out05_img;
    medianBlur(border_extend_img, extend_out05_img,  5);
    Mat extend_out11_img;
    medianBlur(border_extend_img, extend_out11_img, 11);
    Mat extend_out21_img;
    medianBlur(border_extend_img, extend_out21_img, 21);

    // 境界処理用に拡張した部分をトリミング
    const Mat out05_img(extend_out05_img, Rect(EXPIXEL, EXPIXEL, WIDTH, height));
    const Mat out11_img(extend_out11_img, Rect(EXPIXEL, EXPIXEL, WIDTH, height));
    const Mat out21_img(extend_out21_img, Rect(EXPIXEL, EXPIXEL, WIDTH, height));

    // 結果表示
    imshow("out05", out05_img);
    imshow("out11", out11_img);
    imshow("out21", out21_img);

    imwrite("./img/out5.png" , out05_img);
    imwrite("./img/out11.png", out11_img);
    imwrite("./img/out21.png", out21_img);

    // キーボードが押されるまでwait
    waitKey(0);

    return 0;
}