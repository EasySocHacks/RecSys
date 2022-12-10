from typing import List

import pandas as pd

from .base_model import BaseRecModel


class TestModel(BaseRecModel):
    def fit(self, train: pd.DataFrame) -> None:
        pass

    def predict(self, user_id: int, k_recs: int) -> List[int]:
        return list(range(k_recs))
