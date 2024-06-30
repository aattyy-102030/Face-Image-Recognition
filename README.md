# 顔画像認識技術の精度分析
## ■リポジトリ概要
このリポジトリでは、複数の顔検出モデルのバイアスを分析する前段階として、画像認識モデル"EfficientNet-B7"で学習したモデルを、顔検出モデル"[RetinaFace](https://github.com/serengil/retinaface)"に組み込み、顔画像を検出する精度を分析する。

## ■目的と目標
複数の顔検出モデルのバイアスを分析する前段階として、単一のモデルでのフローを実装・理解することを目的としている。

## ■使用モデル
- [EfficientNet-B7](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
- [RetinaFace](https://github.com/serengil/retinaface)

## ■使用データセット
- [man-woman-detection](https://www.kaggle.com/datasets/gmlmrinalini/manwomandetection) (kaggleより参照)

## ■各ファイル概要
- Create_LearnedModel.py
  - データの前処理
  - モデルの学習
  - モデルの保存
  - モデルのテスト
- Predict_RetinaFace.py
  - RetinaFaceでの顔検出
  - 検出した顔を, "Create_LearnedModel.py"で作成したモデルで分類
  - 分類結果を画像として出力
- requirements.txt
  - 使用するパッケージ一覧
- Result.md
  - 構築したモデルの解説, 検証結果

## ■実行手順
**1. データのダウンロード**<br>
kaggleで公開されている[ man-woman-detection ](https://www.kaggle.com/datasets/gmlmrinalini/manwomandetection)をダウンロードしてください。<br>
**2. パッケージのインストール** <br>
以下のコマンドを実行し、必要なパッケージをインストールしてください。
  - pipの場合
  ```
  pip install -r requirements.txt
  ```
  - condaの場合
  ```
  conda install -r requirements.txt
  ```

**3. "Predict_RetinaFace.py"の実行**<br>
**4. 結果の確認**<br>
