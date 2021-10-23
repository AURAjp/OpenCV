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

    const int width = in.cols;
    const int height = in.rows;

    // 周辺領域のピクセルを格納する配列
    // TODO: 暫定で3*3フィルタ
    int pixel_around[9];

    /**
    * 画像データの(1, 1) -> (width - 1, height - 1) までループ.<br>
    * (0, 0)や(width, height)などの境界値は最終的にトリミングするため無視する.
    */
    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            int count = 0;
            // (x - 1, y - 1) -> (x + 1, y + 1)の範囲にフィルターをかける
            for (int py = y - 1; py <= y + 1; py++)
            {
                for (int px = x - 1; px <= x + 1; px++)
                {
                    // 処理対象ピクセルデータの周辺領域のピクセルデータを保存
                    pixel_around[count] = in.at<unsigned char>(py, px);
                    count++;
                }
            }
            // 保存したピクセルデータをソートする
        }
    }
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

    // 境界処理用に画像の上下左右に1ピクセルを反転させて追加する
    constexpr int EXPIXEL = 1;
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

    Mat out03_img;
    medianBlur(resize_img, out03_img, 3);
    Mat extend_out03_img;
    my_medianBlur(border_extend_img, extend_out03_img, 3);
    ///const Mat my_out03_img(extend_out03_img, Rect(EXPIXEL, EXPIXEL, WIDTH, height));

    imshow("out03"  , out03_img);
    //imshow("myout03", my_out03_img);

    // 結果表示
    //imshow("out05", out05_img);
    //imshow("out11", out11_img);
    //imshow("out21", out21_img);

    //imwrite("./img/out5.png" , out05_img);
    //imwrite("./img/out11.png", out11_img);
    //imwrite("./img/out21.png", out21_img);

    // キーボードが押されるまでwait
    waitKey(0);

    return 0;
}