# 顔画像認識技術の精度分析
## ■リポジトリ概要
このリポジトリでは、複数の顔検出モデルのバイアスを分析する前段階として、画像認識モデル"EfficientNet-B7"で学習したモデルを、顔検出モデル"[RetinaFace](https://github.com/serengil/retinaface)"に組み込み、顔画像を検出する精度を分析する。

## ■目的と目標
複数の顔検出モデルのバイアスを分析する前段階として、単一のモデルでのフローを実装・理解することを目的としている。

## ■使用モデル
- [EfficientNet-B7](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
- [RetinaFace](https://github.com/serengil/retinaface)

## ■使用データセット
- [ ] (置き場検討)

## ■各ファイル概要
- Create_LearnedModel.py
  <br>
  - データの前処理
  - モデルの学習
  - モデルの保存
  - モデルのテスト
- model.pth
  - 学習済みモデル
- Result(.md)
  - 構築したモデルの解説, 検証結果

## ■実行手順
**1. データのダウンロード**<br>
※置き場所検討<br>
**2. "Create_LearnedModel.py"の実行**<br>
**3. 結果の確認**<br>

## 今後の展望
