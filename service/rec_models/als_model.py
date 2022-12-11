from typing import List, Optional

import pandas as pd
from implicit.als import AlternatingLeastSquares
from rectools.columns import Columns
from rectools.dataset import Dataset
from rectools.models import ImplicitALSWrapperModel

from .base_model import BaseRecModel

_MODEL_NAME = "ALS"


class ALSFeatures:
    def __init__(
        self,
        user_features_df: pd.DataFrame,
        cat_user_features: List[str],
        item_features_df: pd.DataFrame,
        cat_item_features: List[str],
    ):
        self.user_features_df: pd.DataFrame = user_features_df
        self.cat_user_features: List[str] = cat_user_features
        self.item_features_df: pd.DataFrame = item_features_df
        self.cat_item_features: List[str] = cat_item_features


class ALSModel(BaseRecModel):
    def __init__(
        self,
        train: pd.DataFrame,
        n_factors: int,
        is_fitting_features: bool,
        features: ALSFeatures,
        n_threads: int = 1,
        seed: int = 0,
    ):
        self._n_factors: int = n_factors
        self._is_fitting_features: bool = is_fitting_features
        self._features: ALSFeatures = features
        self._n_threads: int = n_threads
        self._seed: int = seed

        self._dataset: Optional[Dataset] = None

        self.model = ImplicitALSWrapperModel(
            model=AlternatingLeastSquares(
                factors=self._n_factors,
                random_state=self._seed,
                num_threads=self._n_threads,
            ),
            fit_features_together=self._is_fitting_features,
        )

        self.fit(train)

    def recommend(self, user_id: int, k_recs: int) -> List[int]:
        return self.model.recommend(
            users=[user_id],
            dataset=self._dataset,
            k=k_recs,
            filter_viewed=True,
        )[Columns.Item]

    def fit(
        self,
        train: pd.DataFrame
    ) -> None:
        self._dataset = Dataset.construct(
            interactions_df=train,
            user_features_df=self._features.user_features_df,
            cat_user_features=["sex", "age", "income"],
            item_features_df=self._features.item_features_df,
            cat_item_features=["genre", "content_type"],
        )

        self.model.fit(self._dataset)
