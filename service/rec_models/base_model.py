from abc import ABC
from typing import List

import pandas as pd


class BaseRecModel(ABC):
    def fit(self, train: pd.DataFrame) -> None:
        pass

    def predict(self, user_id: int, k_recs: int) -> List[int]:
        pass
