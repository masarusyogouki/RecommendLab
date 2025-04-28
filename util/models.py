import dataclasses
import pandas as pd
from typing import List, Dict

# 推薦システムの学習と評価に使うデータセット
@dataclasses.dataclass(frozen=True)
class Dataset:
    # 学習用の評価値データセット
    train: pd.DataFrame
    # テスト用の評価値データセット
    test: pd.DataFrame
    # ランキング指標のテストデータセット。キーはユーザーID、バリューはユーザーが高評価したアイテムIDのリスト。
    test_user2items: Dict[int, List[int]]
    # アイテムのコンテンツ情報
    item_content: pd.DataFrame

@dataclasses.dataclass(frozen=True)
class RecommendResult:
    # テストデータセットの予測評価値。RSMEの評価
    rating: pd.DataFrame
    # キーはユーザーID, バリューはおすすめアイテムのIDのリスト。ランキング指標の評価
    user2items: Dict[int, List[int]]