import os
from typing import Any, Dict, List, Optional, Tuple, Union

import dill
import hnswlib
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


class NeighParams:
    def __init__(
        self,
        M: int = 128,
        efC: int = 256,
        efS: int = 256,
        threads: int = 16
    ):
        self.M = M
        self.efC = efC
        self.efS = efS
        self.threads = threads


class FactoringMachineModel(BaseRecModel):
    def __init__(
        self,
        train: pd.DataFrame,
        hyper_params: FMHyperParams,
        tune_params: FMTuneParams,
        neigh_params: NeighParams,
        features: FMFeatures,
        save=False,
    ):
        self.neigh_params = neigh_params
        self.features: FMFeatures = features

        self.embeddings: Optional[FMEmbedding] = None
        self.mappings: Optional[Mappings] = None
        self.label: Any = None
        self.distance: Any = None

        self.model = LightFMWrapperModel(
            LightFM(
                no_components=hyper_params.n_factors,
                loss=hyper_params.loss,
                random_state=tune_params.seed,
                learning_rate=hyper_params.lr,
                user_alpha=hyper_params.ua,
                item_alpha=hyper_params.ia,
            ),
            epochs=tune_params.n_epoch,
            num_threads=tune_params.n_threads,
        )

        self.fit(
            train=train,
            save=save,
        )

    @staticmethod
    def _augment_inner_product(factors: int) -> Tuple[float, np.ndarray]:
        normed_factors = np.linalg.norm(factors, axis=1)
        max_norm = normed_factors.max()

        extra_dim = np.sqrt(max_norm ** 2 - normed_factors ** 2).reshape(-1, 1)
        augmented_factors = np.append(factors, extra_dim, axis=1)

        return max_norm, augmented_factors

    def recommend(
        self,
        user_id: int,
        k_recs: int,
    ) -> List[int]:
        if user_id not in self.mappings.users_mapping:
            return [
                self.mappings.items_inv_mapping[item]
                for item
                in np.random.randint(
                    low=0,
                    high=self.embeddings.item_embeddings.shape[0],
                    size=k_recs,
                )
            ]

        return [
            self.mappings.items_inv_mapping[item]
            for item
            in list(self.label[self.mappings.users_mapping[user_id], :])
        ]

    @staticmethod
    def augment_inner_product(factors: np.ndarray) -> Union[Any, np.ndarray]:
        normed_factors = np.linalg.norm(factors, axis=1)
        max_norm = normed_factors.max()

        extra_dim = np.sqrt(max_norm ** 2 - normed_factors ** 2).reshape(-1, 1)
        augmented_factors = np.append(factors, extra_dim, axis=1)
        return max_norm, augmented_factors

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
        save=False,
    ) -> None:
        rectools_dataset = Dataset.construct(
            interactions_df=train,
            user_features_df=self.features.user_features_df,
            cat_user_features=self.features.cat_user_features,
            item_features_df=self.features.item_features_df,
            cat_item_features=self.features.cat_item_features,
        )

        self.get_mappings(train)

        self.model.fit(rectools_dataset)

        user_embeddings, item_embeddings = self.model.get_vectors(
            rectools_dataset
        )

        _, augmented_item_embeddings = self.augment_inner_product(
            item_embeddings
        )
        extra_zero = np.zeros((user_embeddings.shape[0], 1))
        augmented_user_embeddings = np.append(
            user_embeddings,
            extra_zero,
            axis=1
        )

        max_elements, dim = augmented_item_embeddings.shape
        hnsw = hnswlib.Index("ip", dim)
        hnsw.init_index(
            max_elements,
            self.neigh_params.M,
            self.neigh_params.efC
        )
        hnsw.add_items(augmented_item_embeddings)
        hnsw.set_ef(self.neigh_params.efS)

        self.label, distance = hnsw.knn_query(
            augmented_user_embeddings,
            k=10
        )
        self.distance = 1 - distance

        self.embeddings = FMEmbedding(
            augmented_user_embeddings,
            augmented_item_embeddings
        )

        if save:
            with open(f'{ROOT_DIR}/dumps/fm.dill', 'wb') as f:
                dill.dump(self, f)

    @staticmethod
    def load() -> 'FactoringMachineModel':
        with open(f'{ROOT_DIR}/dumps/fm.dill', 'rb') as f:
            return dill.load(f)
