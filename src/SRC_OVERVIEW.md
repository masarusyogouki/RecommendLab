# srcディレクトリ概要

## 目次
1. [概要](#概要)  
2. [ディレクトリ構成](#ディレクトリ構成)  
   - [`base_recommender.py`](#baserecommenderpy) 
   - [`random.py`](#randompy) 
   - [`popularity.py`](#popularitypy) 

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

### random.py

- **継承**: `BaseRecommender`  
- **役割**: ランダムな予測評価値とランキングを返すベースライン実装

---

#### `recommend(dataset: Dataset, **kwargs) -> RecommendResult` の処理フロー

1. **ID→インデックスの割り当て**  
   - 学習データのユーザーID・映画IDをユニーク取得  
   - それぞれを 0 始まりの連続インデックスにマッピング  

2. **予測評価値行列の生成**  
   - `(ユーザー数 × 映画数)` の形状で一様乱数（0.5～5.0）を生成  

3. **テストセット用予測スコア取得**  
   - テストデータの各行について、対応する乱数を取り出し  
   - 学習時に未登場の映画IDはその都度新規乱数を生成  
   - `rating_pred` カラムとして DataFrame に追加  

4. **ランキング推薦リストの作成**  
   - 各ユーザーの評価済み映画を除外  
   - 乱数行列をスコア降順にソートし、未評価映画から上位10作品を選定  
   - ユーザーID → 推薦映画IDリスト の `dict` を構築 (`pred_user2items`)  

5. **結果を返却**  
   - `RecommendResult(rating_pred_series, pred_user2items)` を返す  

### popularity.py
