# utilディレクトリ概要

## 目次
1. [概要](#概要)  
2. [ディレクトリ構成](#ディレクトリ構成)  
   - [`data_loader.py`](#data_loaderpy)  
   - [`metric_calculator.py`](#metric_calculatorpy)  
   - [`models.py`](#modelspy)  

## 概要
`utils/`ディレクトリには、プロジェクト全体で共通して利用するロジックをまとめている。<br>主な役割は以下の通りである。

## ディレクトリ構成
### data_loader.py
`DataLoader`クラスはMovieLensデータセットを読み込み、学習用・評価用に分割したうえで、推薦システムに必要なフォーマットに整形して返す。

1. **クラス定義**
```python
class DataLoader:
    def __init__(self, num_users: int = 1000, num_test_items: int = 5, data_path: str = "../data/ml-10M100K/"):
        self.num_users = num_users
        self.num_test_items = num_test_items
        self.data_path = data_path
```
- `num_users`: 使用するユーザー数(1000ユーザーに限定することで実験を高速化)
- `num_test_items`: 各ユーザーからテスト用に抽出する最新評価件数(最新の5件をテスト用に設定)
- `data_path`: MovieLensデータファイルが格納されたディレクトリパス

2. **load()メソッド**
```python
    def load(self) -> Dataset:
        ratings, movie_content = self._load()
        movielens_train, movielens_test = self._split_data(ratings)
        # ranking用の評価データは、各ユーザーの評価値が4以上の映画だけを正解とする
        # キーはユーザーID、バリューはユーザーが高評価したアイテムIDのリスト
        movielens_test_user2items = (
            movielens_test[movielens_test.rating >= 4].groupby("user_id").agg({"movie_id": list})["movie_id"].to_dict()
        )
        return Dataset(movielens_train, movielens_test, movielens_test_user2items, movie_content)
```
- _load()で映画情報・評価情報を読み込み
- _split_data()で学習用・テスト用に分割
- テスト用データから「評価値≥ 4 の映画リスト」をユーザごとに抽出

3. **_load()プライベートメソッド**
```python
    def _load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # 映画の情報の読み込み(10197作品)
        # movie_idとタイトル名のみ使用
        m_cols = ["movie_id", "title", "genre"]
        movies = pd.read_csv(
            os.path.join(self.data_path, "movies.dat"),
            names=m_cols,
            sep="::",
            encoding="latin-1",
            engine="python"
        )
        # genreをlist形式で保持する
        movies["genre"] = movies.genre.apply(lambda x: x.split("|"))
```
- 映画情報読み込み
    - `movies.dat`を読み込み、`movie_id`・`title`・`genre`を抽出
    - `genre`を `["Action","Comedy",…]` の list に変換

```python
        # ユーザが付与した映画のタグ情報の読み込み
        t_cols = ["user_id", "movie_id", "tag", "timestamp"]
        user_tagged_movies = pd.read_csv(
            os.path.join(self.data_path, "tags.dat"), 
            names=t_cols, 
            sep="::", 
            engine="python"
        )
        # tagをすべて小文字化して表記ゆれをなくす
        user_tagged_movies["tag"] = user_tagged_movies["tag"].str.lower()
        # movie_idごとにtagをlist形式で保持する
        movie_tags = user_tagged_movies.groupby("movie_id").agg({"tag": list})

        # タグ情報を結合する
        movies = movies.merge(movie_tags, on="movie_id", how="left")
```
- タグ情報読み込み
    - `tags.dat`から`user_id`・`movie_id`・`tag`・`timestamp`を読み込み
    - `tag`を小文字化し、映画ごとにタグリストを集計
    - 映画情報とマージして`movie_content`を作成

```python
        # 評価データの読み込み
        r_cols = ["user_id", "movie_id", "rating", "timestamp"]
        ratings = pd.read_csv(
            os.path.join(self.data_path, "ratings.dat"), 
            names=r_cols, 
            sep="::", 
            engine="python"
        )

        # 実験を高速で行うためにユーザ数を制限する
        valid_user_ids = sorted(ratings.user_id.unique())[: self.num_users]
        ratings = ratings[ratings.user_id <= max(valid_user_ids)]

        # 上記のデータを結合する
        movielens_ratings = ratings.merge(movies, on="movie_id")

        return movielens_ratings, movies
```
- 評価情報読み込み
    - `ratings.dat`から`user_id`・`movie_id`・`tag`・`timestamp`を取得
    - 映画メタ情報とマージ

3. **_split_data()プライベートメソッド**
```python
    def _split_data(self, movielens: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        # 学習用とテスト用にデータを分割する
        # 各ユーザの直近の５件の映画を評価用に使い、それ以外を学習用とする
        # まずは、それぞれのユーザが評価した映画の順序を計算する
        # 直近付与した映画から順番を付与していく(0始まり)
        movielens["rating_order"] = movielens.groupby("user_id")["timestamp"].rank(ascending=False, method="first")
        movielens_train = movielens[movielens["rating_order"] > self.num_test_items]
        movielens_test = movielens[movielens["rating_order"] <= self.num_test_items]
        return movielens_train, movielens_test
```
- 各ユーザごとに評価日時の降順で順位を付与 (`rating_order`)
- 上位 `num_test_items` 件をテスト用、それ以外を学習用に分割
