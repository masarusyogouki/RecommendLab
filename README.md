# RecommendLab

## 概要
このリポジトリは推薦システム学習用です。

## コード構成
```text
RecommendLab/
├── data/                        # データセットディレクトリ
├── notebooks/                   # 推薦アルゴリズム実行用ノートブック
│   ├── download_movielens.ipynb # MovieLensのダウンロード
│   └── Random.ipynb
├── src/
│   ├── base_recommender.py      # アルゴリズムのインターフェース
│   └── Random.py                # ランダム推薦アルゴリズム
├── util/                        # 共通ロジック
│   ├── data_loader.py
│   ├── metric_calculator.py     # 評価指標計算モジュール
│   └── models.py                # データコンテナ
└── .gitignore
```