import os
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
from lightfm import LightFM
from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import LightFMWrapperModel

from service.rec_models import BaseRecModel
from service.rec_models.exceptions import RecModelNotLearnedYetException

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


class FactoringMachineModel(BaseRecModel):
    def __init__(
        self,
        dataset: Union[Dataset, pd.DataFrame],
        hyper_params: FMHyperParams,
        tune_params: FMTuneParams,
    ):
        self.dataset = dataset

        self._hyper_params = hyper_params
        self._tune_params = tune_params

        self.user_embeddings: Any = None
        self.item_embeddings: Any = None

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

        self.fit(self.dataset)

    @staticmethod
    def prepare():
        os.environ["OPENBLAS_NUM_THREADS"] = "1"

        interactions = pd.read_csv(f'{ROOT_DIR}/kion_train/interactions.csv')
        users = pd.read_csv(f'{ROOT_DIR}/kion_train/users.csv')
        items = pd.read_csv(f'{ROOT_DIR}/kion_train/items.csv')

        cold_users = list(set(users["user_id"]) - set(interactions["user_id"]))
        cold_users.append(0)
        data = {
            "user_id": cold_users,
            "item_id": [0] * len(cold_users),
            "last_watch_dt": "1990-01-01",
            "total_dur": [0] * len(cold_users),
            "watched_pct": [0] * len(cold_users)
        }
        new_interactions = pd.DataFrame(data)
        interactions = interactions.append(new_interactions)

        cold_items = list(set(items["item_id"]) - set(interactions["item_id"]))
        cold_items.append(0)
        data = {
            "user_id": [0] * len(cold_items),
            "item_id": cold_items,
            "last_watch_dt": "1990-01-01",
            "total_dur": [0] * len(cold_items),
            "watched_pct": [0] * len(cold_items)
        }
        new_interactions = pd.DataFrame(data)
        interactions = interactions.append(new_interactions)

        interactions[Columns.Weight] = \
            np.where(interactions['watched_pct'] > 10, 3, 1)
        interactions[Columns.Weight].value_counts(normalize=True)

        Columns.Datetime = 'last_watch_dt'

        interactions[Columns.Datetime] = pd.to_datetime(
            interactions[Columns.Datetime], format='%Y-%m-%d')

        user_features_frames = []
        for feature in ["sex", "age", "income"]:
            feature_frame = users.reindex(columns=[Columns.User, feature])
            feature_frame.columns = ["id", "value"]
            feature_frame["feature"] = feature
            user_features_frames.append(feature_frame)
        user_features = pd.concat(user_features_frames)

        items["genre"] = items["genres"].str.lower() \
            .str.replace(", ", ",", regex=False).str.split(",")
        genre_feature = items[["item_id", "genre"]].explode("genre")
        genre_feature.columns = ["id", "value"]
        genre_feature["feature"] = "genre"

        content_feature = items.reindex(columns=[Columns.Item, "content_type"])
        content_feature.columns = ["id", "value"]
        content_feature["feature"] = "content_type"

        item_features = pd.concat((genre_feature, content_feature))

        dataset = Dataset.construct(
            interactions_df=interactions,
            user_features_df=user_features,
            cat_user_features=["sex", "age", "income"],
            item_features_df=item_features,
            cat_item_features=["genre", "content_type"],
        )

        return FactoringMachineModel(
            dataset,
            FMHyperParams(
                64,
                "warp",
                0.01,
                0,
                0,
            ),
            FMTuneParams(
                1,
                16,
                123,
            ),
        )

    @staticmethod
    def _augment_inner_product(factors: int) -> Tuple[float, np.ndarray]:
        normed_factors = np.linalg.norm(factors, axis=1)
        max_norm = normed_factors.max()

        extra_dim = np.sqrt(max_norm ** 2 - normed_factors ** 2).reshape(-1, 1)
        augmented_factors = np.append(factors, extra_dim, axis=1)

        return max_norm, augmented_factors

    def recommend(self, user_id: int, k_recs: int) -> List[int]:
        labels, _ = self._recommend_all(
            self.user_embeddings[[user_id], :],
            self.item_embeddings
        )

        return list(labels.flatten())

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

    def fit(
        self,
        dataset: Union[Dataset, pd.DataFrame],
    ) -> None:
        self.model.fit(dataset)

        if dataset is None:
            raise RecModelNotLearnedYetException(_MODEL_NAME)

        self.user_embeddings, self.item_embeddings = self.model.get_vectors(
            dataset
        )
