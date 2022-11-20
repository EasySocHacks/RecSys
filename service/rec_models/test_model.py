from typing import List

from .base_model import BaseRecModel


class TestModel(BaseRecModel):
    def recommend(self, user_id: int, k_recs: int) -> List[int]:
        return list(range(k_recs))
