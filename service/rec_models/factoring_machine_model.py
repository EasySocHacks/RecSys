import os
from typing import Any, Dict, List, Optional, Tuple

import dill
import numpy as np
import pandas as pd
from lightfm import LightFM
from rectools.dataset import Dataset
from rectools.models import LightFMWrapperModel

from service.rec_models import BaseRecModel

_MODEL_NAME = "FactoringMachine"
ROOT_DIR = os.path.abspath(os.curdir)


class FMHyperParams:
    def __init__(
        self,
        n_factors: int,
        loss: str,
        lr: float,
        ua: int,
        ia: int,
    ):
        self.n_factors: int = n_factors
        self.loss: str = loss
        self.lr: float = lr
        self.ua: int = ua
        self.ia: int = ia


class FMTuneParams:
    def __init__(
        self,
        n_epoch: int = 1,
        n_threads: int = 1,
        seed: int = 0,
    ):
        self.n_epoch: int = n_epoch
        self.n_threads: int = n_threads
        self.seed: int = seed


class FMFeatures:
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


class Mappings:
    def __init__(
        self,
        users_inv_mapping: Dict[int, int],
        users_mapping: Dict[int, int],
        items_inv_mapping: Dict[int, int],
        items_mapping: Dict[int, int],
    ) -> None:
        self.users_inv_mapping: Dict[int, int] = users_inv_mapping
        self.users_mapping: Dict[int, int] = users_mapping
        self.items_inv_mapping: Dict[int, int] = items_inv_mapping
        self.items_mapping: Dict[int, int] = items_mapping


class FMEmbedding:
    def __init__(
        self,
        user_embeddings: Any,
        item_embeddings: Any,
    ):
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings


class FactoringMachineModel(BaseRecModel):
    def __init__(
        self,
        train: pd.DataFrame,
        hyper_params: FMHyperParams,
        tune_params: FMTuneParams,
        features: FMFeatures,
        save=False,
    ):
        self._hyper_params: FMHyperParams = hyper_params
        self._tune_params: FMTuneParams = tune_params

        self.features: FMFeatures = features

        self.embeddings: Optional[FMEmbedding] = None

        self.mappings: Optional[Mappings] = None

        self.user_mapping: Dict[int, int] = {}

        self.model = LightFMWrapperModel(
            LightFM(
                no_components=self._hyper_params.n_factors,
                loss=self._hyper_params.loss,
                random_state=self._tune_params.seed,
                learning_rate=self._hyper_params.lr,
                user_alpha=self._hyper_params.ua,
                item_alpha=self._hyper_params.ia,
            ),
            epochs=self._tune_params.n_epoch,
            num_threads=self._tune_params.n_threads,
        )

        self.fit(
            train=train,
            features=features,
            save=save,
        )

    @staticmethod
    def _augment_inner_product(factors: int) -> Tuple[float, np.ndarray]:
        normed_factors = np.linalg.norm(factors, axis=1)
        max_norm = normed_factors.max()

        extra_dim = np.sqrt(max_norm ** 2 - normed_factors ** 2).reshape(-1, 1)
        augmented_factors = np.append(factors, extra_dim, axis=1)

        return max_norm, augmented_factors

    def recommend(self, user_id: int, k_recs: int) -> List[int]:
        try:
            if user_id not in self.mappings.users_mapping:
                return list(range(10))

            labels, _ = self._recommend_all(
                self.embeddings.user_embeddings[[self.mappings.users_mapping[
                                                     user_id]
                                                 ], :],
                self.embeddings.item_embeddings
            )

            return list(labels.flatten())
        except Exception:
            return list(range(10))

    @staticmethod
    def _recommend_all(
        query_factors: np.ndarray,
        index_factors: np.ndarray,
        topn: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        output = query_factors.dot(index_factors.T)
        argpartition_indices = np.argpartition(output, -topn)[:, -topn:]

        x_indices = np.repeat(np.arange(output.shape[0]), topn)
        y_indices = argpartition_indices.flatten()
        top_value = output[x_indices, y_indices].reshape(output.shape[0], topn)
        top_indices = np.argsort(top_value)[:, ::-1]

        y_indices = top_indices.flatten()
        top_indices = argpartition_indices[x_indices, y_indices]
        labels = top_indices.reshape(-1, topn)
        distances = output[x_indices, top_indices].reshape(-1, topn)
        return labels, distances

    def get_mappings(self, train) -> None:
        users_inv_mapping = dict(enumerate(train['user_id'].unique()))
        users_mapping = {v: k for k, v in users_inv_mapping.items()}

        items_inv_mapping = dict(enumerate(train['item_id'].unique()))
        items_mapping = {v: k for k, v in items_inv_mapping.items()}
        self.mappings = Mappings(
            users_inv_mapping, users_mapping, items_inv_mapping, items_mapping,
        )

    def fit(
        self,
        train: pd.DataFrame,
        features: FMFeatures,
        save=False,
    ) -> None:
        rectools_dataset = Dataset.construct(
            interactions_df=train,
            user_features_df=features.user_features_df,
            cat_user_features=features.cat_user_features,
            item_features_df=features.item_features_df,
            cat_item_features=features.cat_item_features,
        )

        self.get_mappings(train)

        self.model.fit(rectools_dataset)

        user_embeddings, item_embeddings = self.model.get_vectors(
            rectools_dataset
        )

        self.embeddings = FMEmbedding(user_embeddings, item_embeddings)

        if save:
            with open(f'{ROOT_DIR}/dumps/fm.dill', 'wb') as f:
                dill.dump(self, f)

    @staticmethod
    def load() -> 'FactoringMachineModel':
        with open(f'{ROOT_DIR}/dumps/fm.dill', 'rb') as f:
            return dill.load(f)
