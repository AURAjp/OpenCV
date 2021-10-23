#include <opencv2/opencv.hpp>
#include <vector>

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
        out = in.clone();
        return;
    }
    // フィルタサイズが奇数か、入力画像がCV_8UC1かどうかチェック
    CV_Assert(filter_size % 2 == 1 && in.type() == 0);

    // フィルタが影響を与えるのは処理対象ピクセルデータから前後半分のエリア
    const int affect_area = (filter_size - 1) / 2;

    // 境界処理用に画像の上下左右にフィルタが影響を与えるピクセル分を反転させて追加する
    Mat border_extend_img;
    copyMakeBorder(in, border_extend_img, affect_area, affect_area, affect_area, affect_area, BORDER_REFLECT_101);

    // 処理範囲を特定する
    const int width = border_extend_img.cols;
    const int height = border_extend_img.rows;

    // 周辺領域のピクセルを格納する動的配列
    vector<int> pixel_around(pow(filter_size, 2), 0);
    // 出力用画像領域の確保
    Mat dst_img = Mat(height, width, CV_8UC1);

    /**
    * 処理対象ピクセルデータ(x, y)について、
    * (affect_area, affect_area) -> (width - affect_area, height - affect_area) までループ.<br>
    * (0, 0)や(width, height)などの境界値は最終的にトリミングするため無視する.
    */
    for (int y = affect_area; y < height - affect_area; y++)
    {
        for (int x = affect_area; x < width - affect_area; x++)
        {
            // 配列の添字
            int index = 0;
            // (x - affect_area, y - affect_area) -> (x + affect_area, y + affect_area)の範囲にフィルターをかける
            for (int py = y - affect_area; py <= y + affect_area; py++)
            {
                for (int px = x - affect_area; px <= x + affect_area; px++)
                {
                    // 処理対象ピクセルデータの周辺領域のピクセルデータを保存
                    pixel_around[index] = border_extend_img.at<unsigned char>(py, px);
                    index++;
                }
            }
            // 保存したピクセルデータをソートして中央値を取得
            sort(begin(pixel_around), end(pixel_around));
            int median = pixel_around[pixel_around.size() / 2 + 1];
            // 取得した中央値を処理対象ピクセルに上書きする
            dst_img.at<unsigned char>(y, x) = median;
        }
    }
    // 境界処理用に拡張した部分をトリミング
    out = Mat(dst_img, Rect(affect_area, affect_area, width - affect_area * 2, height - affect_area * 2));
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

    // Median Blur
    Mat out05_img;
    my_medianBlur(resize_img, out05_img,  5);
    Mat out11_img;
    my_medianBlur(resize_img, out11_img, 11);
    Mat out21_img;
    my_medianBlur(resize_img, out21_img, 21);

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