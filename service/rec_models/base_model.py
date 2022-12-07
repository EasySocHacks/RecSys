from abc import ABC
from typing import List, Union

import pandas as pd
from rectools.dataset import Dataset


class BaseRecModel(ABC):
    def recommend(self, user_id: int, k_recs: int) -> List[int]:
        pass

    def learn(self, dataset: Union[Dataset, pd.DataFrame]) -> None:
        pass
