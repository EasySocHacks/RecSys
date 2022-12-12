from abc import ABC
from typing import List

import pandas as pd


class BaseRecModel(ABC):
    def recommend(self, user_id: int, k_recs: int) -> List[int]:
        pass

    def fit(self, train: pd.DataFrame) -> None:
        pass
