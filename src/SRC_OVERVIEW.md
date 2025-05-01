# srcディレクトリ概要

## 目次
1. [概要](#概要)  
2. [ディレクトリ構成](#ディレクトリ構成)  
   - [`base_recommender.py`](#baserecommenderpy) 
   - [`random.py`](#random) 

## 概要

## ディレクトリ構成
### base_recommender.py
MovieLensデータセットを使った推薦アルゴリズムを実装・評価するための抽象ベースクラス
- **抽象メソッド `recommend()`**  
  - 推薦ロジックのコア  
  - 引数: `Dataset`  
  - 返り値: `RecommendResult`（予測スコア・推薦アイテム一覧）

- **共通処理 `run_sample()`**  
  1. `DataLoader` を使って学習用／テスト用データを読み込む  
  2. `recommend()` を呼び出して予測結果を取得  
  3. `MetricCalculator` で RMSE・Precision@10・Recall@10 を一括計算し、結果をコンソールに出力

- **拡張性**  
  - `recommend()` を実装したサブクラスを作るだけで、データ取得から評価まで一連の流れを簡単に動かせる  
  - `DataLoader` のパラメータ（`num_users`, `num_test_items`, `data_path` など）を柔軟に指定可能
