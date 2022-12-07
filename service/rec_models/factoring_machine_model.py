import os
from typing import List, Union

import dill
import numpy as np
import pandas as pd
from lightfm import LightFM
from rectools.dataset import Dataset
from rectools.models import LightFMWrapperModel

from service.rec_models import BaseRecModel
from service.rec_models.exceptions import RecModelNotLearnedYetException

_MODEL_NAME = "FactoringMachine"


class FactoringMachineModel(BaseRecModel):
    def __init__(
        self,
        dataset,
        n_factors,
        loss,
        lr,
        ua,
        ia,
        n_epoch=1,
        n_threads=1,
        seed=0,
        save=False,
    ):
        self._n_factors = n_factors
        self._loss = loss
        self._lr = lr
        self._ua = ua
        self._ia = ia
        self._n_epoch = n_epoch
        self._n_threads = n_threads
        self._seed = seed

        self._dataset = dataset

        self._M = 48
        self._efC = 100
        self._efS = 100
        self._space_name = "negdotprod"

        self.hnsw = None
        self.query_matrix = None

        self.user_embeddings = None
        self.item_embeddings = None

        self.model = LightFMWrapperModel(
            LightFM(
                no_components=self._n_factors,
                loss=self._loss,
                random_state=self._seed,
                learning_rate=self._lr,
                user_alpha=self._ua,
                item_alpha=self._ia,
            ),
            epochs=self._n_epoch,
            num_threads=self._n_threads,
        )

        self.fit(dataset, save)

    @staticmethod
    def _augment_inner_product(factors):
        normed_factors = np.linalg.norm(factors, axis=1)
        max_norm = normed_factors.max()

        extra_dim = np.sqrt(max_norm ** 2 - normed_factors ** 2).reshape(-1, 1)
        augmented_factors = np.append(factors, extra_dim, axis=1)

        return max_norm, augmented_factors

    def recommend(self, user_id: int, k_recs: int) -> List[int]:
        return self._recommend_all(
            self.user_embeddings[[user_id], :],
            self.item_embeddings
        )[user_id].flatten()

    @staticmethod
    def _recommend_all(query_factors, index_factors, topn=10):
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
        save: bool = False,
    ) -> None:
        self._dataset = dataset

        self.model.fit(dataset)

        if self._dataset is None:
            raise RecModelNotLearnedYetException(_MODEL_NAME)

        self.user_embeddings, self.item_embeddings = self.model.get_vectors(
            self._dataset
        )

        if save:
            if not os.path.exists("dumps"):
                os.makedirs("dumps")
            with open('dumps/factoring_machine_model.dill', 'wb') as f:
                dill.dump(self, f)

    @staticmethod
    def load(
        filename: str = "factoring_machine_model.dill"
    ) -> "FactoringMachineModel":
        with open("dumps/" + filename, 'rb') as f:
            return dill.load(f)
