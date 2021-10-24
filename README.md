# 概要

OpenCV

## 課題1

- 画像のリサイズ
- 市松模様の生成

## 課題2

- filter2Dでブラーをかける

## 課題3

- Canny法によるエッジ検出
- Bilateralフィルタによる画像平滑化
- ビット演算による画像合成

## 課題4

- DFT
- IDFT
- ローパス

## 課題5

- MedianFilterを自作

# 環境構築

## 動作環境

- OpenCV 2.4.13
- C++ 20

## 開発環境

- Visual Studio 2019

## ビルド

### 1. Pathを通す
https://brain.cc.kogakuin.ac.jp/~kanamaru/lecture/opencv/index1.html

### 2. プロジェクトの設定

1. プロジェクト > プロパティ > VC++ ディレクトリと進む
2. インクルード ディレクトリに`C:\opencv\build\include`を追加
3. ライブラリ ディレクトリに`C:\opencv\build\x64\vc14\lib`を追加
