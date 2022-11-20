from abc import ABC
from typing import List


class BaseRecModel(ABC):
    def recommend(self, user_id: int, k_recs: int) -> List[int]:
        pass
