from typing import List, Union

import pandas as pd
from implicit.als import AlternatingLeastSquares
from rectools.columns import Columns
from rectools.dataset import Dataset
from rectools.models import ImplicitALSWrapperModel

from .base_model import BaseRecModel

_MODEL_NAME = "ALS"


class ALSModel(BaseRecModel):
    def __init__(
        self,
        dataset,
        n_factors,
        is_fitting_features,
        n_threads=1,
        seed=0,
    ):
        self._n_factors = n_factors
        self._is_fitting_features = is_fitting_features
        self._n_threads = n_threads
        self._seed = seed

        self._dataset = dataset

        self.model = ImplicitALSWrapperModel(
            model=AlternatingLeastSquares(
                factors=self._n_factors,
                random_state=self._seed,
                num_threads=self._n_threads,
            ),
            fit_features_together=self._is_fitting_features,
        )

        self.fit(dataset)

    def recommend(self, user_id: int, k_recs: int) -> List[int]:
        return self.model.recommend(
            users=[user_id],
            dataset=self._dataset,
            k=k_recs,
            filter_viewed=True,
        )[Columns.Item]

    def fit(
        self,
        dataset: Union[Dataset, pd.DataFrame],
    ) -> None:
        self._dataset = dataset

        self.model.fit(dataset)
