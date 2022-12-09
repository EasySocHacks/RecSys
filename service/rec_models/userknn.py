from collections import Counter
from collections.abc import Mapping
from typing import Callable, Dict, List, Optional, Tuple

import dill
import numpy as np
import pandas as pd
import scipy as sp
from implicit.nearest_neighbours import ItemItemRecommender
from rectools.dataset import Dataset
from rectools.models.popular import PopularModel

from .base_model import BaseRecModel


class Models:
    """Inner Models"""

    def __init__(self,
                 user_knn: ItemItemRecommender,
                 popular: PopularModel
                 ) -> None:
        self.user_knn = user_knn
        self.popular = popular

class Mappings:
    """Ids Mappings for implicit itemKNN"""

    def __init__(self,
                 users_inv_mapping: Dict[int, int],
                 users_mapping: Dict[int, int],
                 items_inv_mapping: Dict[int, int],
                 items_mapping: Dict[int, int],
                 ) -> None:
        self.users_inv_mapping: Dict[int, int] = users_inv_mapping
        self.users_mapping: Dict[int, int] = users_mapping
        self.items_inv_mapping: Dict[int, int] = items_inv_mapping
        self.items_mapping: Dict[int, int] = items_mapping


class Watched:
    """Watched items for single and batch recommendations"""

    def __init__(self, watched: pd.DataFrame):
        self.watched: pd.DataFrame = watched
        self.watched_dict: Mapping = watched['item_id'].to_dict()


class IDF:
    """Calculated items' IDF for single and batch recommendations"""

    def __init__(self, item_idf: pd.DataFrame):
        self.item_idf: pd.DataFrame = item_idf
        self.item_idf_dict: Mapping = \
            item_idf.set_index('index')['idf'].to_dict()


class UserKnn(BaseRecModel):
    """Class for fit-predict UserKNN model
    based on ItemKNN model from implicit.nearest_neighbours.

    Parameters
    ----------
    IIR_model : ItemItemRecommender
        Base itemKNN class
    popularity : str, optional, default `"n_users"`
        Method of calculating item popularity
    N_users : int, optional, default ``50``
        Nearest user count for KNN model
    """

    def __init__(self, IIR_model: ItemItemRecommender,
                 popularity: str = "n_users",
                 N_users: int = 50) -> None:
        self.is_fitted: bool = False

        self.mappings: Optional[Mappings] = None

        self.models = Models(
            IIR_model,
            PopularModel(
                popularity=popularity,
                verbose=0,
            )
        )

        self.dataset: Optional[Dataset] = None
        self.watched: Optional[Watched] = None
        self.item_idf: Optional[IDF] = None
        self.N_users: int = N_users

    def get_mappings(self, train) -> None:
        users_inv_mapping = dict(enumerate(train['user_id'].unique()))
        users_mapping = {v: k for k, v in users_inv_mapping.items()}

        items_inv_mapping = dict(enumerate(train['item_id'].unique()))
        items_mapping = {v: k for k, v in items_inv_mapping.items()}
        self.mappings = Mappings(
            users_inv_mapping,
            users_mapping,
            items_inv_mapping,
            items_mapping
        )

    def get_matrix(self, df: pd.DataFrame,
                   weight_col: str = None) \
            -> sp.sparse.coo_matrix:
        if weight_col:
            weights = df[weight_col].astype(np.float32)
        else:
            weights = np.ones(len(df), dtype=np.float32)

        interaction_matrix = sp.sparse.coo_matrix((
            weights,
            (
                df['user_id'].map(self.mappings.users_mapping.get),
                df['item_id'].map(self.mappings.items_mapping.get)
            )
        ))

        self.watched = Watched(df.groupby('user_id').agg({'item_id': list}))
        return interaction_matrix

    @staticmethod
    def load(filename: str = "userknn.dill") -> "UserKnn":
        with open("dumps/" + filename, 'rb') as f:
            return dill.load(f)

    @staticmethod
    def idf(n: int, x: float) -> float:
        return np.log((1 + n) / (1 + x) + 1)

    def _count_item_idf(self, df: pd.DataFrame, n: int) -> None:
        item_cnt: Counter = Counter(df['item_id'].values)
        item_idf = pd.DataFrame.from_dict(item_cnt, orient='index',
                                          columns=['doc_freq']).reset_index()
        item_idf['idf'] = item_idf['doc_freq'].apply(
            lambda x: self.idf(n, x))
        self.item_idf = IDF(item_idf)

    def fit(self, train: pd.DataFrame,
            save: bool = False) -> None:
        """
        Fitting inner models.

        Parameters
        ----------
        train : pd.DataFrame
            User-Item interactions
        save : bool, optional, default ``False``
            Pickle model to a file if ``True``
        """

        self.get_mappings(train)
        weights_matrix = self.get_matrix(train, weight_col='weight')

        self._count_item_idf(train, train.shape[0])

        self.models.user_knn.fit(weights_matrix)

        self.dataset = Dataset.construct(
            train,
        )
        self.models.popular.fit(self.dataset)

        self.is_fitted = True

        if save:
            with open('dumps/userknn.dill', 'wb') as f:
                dill.dump(self, f)

    def _predict_userknn(self, user_id: int, k_recs: int) -> np.ndarray:
        """
        Make userKNN recommendations for curtain user.

        Parameters
        ----------
        user_id : int
            User ID
        k_recs : int
            Count of items to recommend

        Returns
        ----------
        np.ndarray
            Recommended items' IDs. Can return less than k_recs IDs
            if user's nearest neighbours have few matched items.
        """

        if user_id not in self.mappings.users_mapping:
            return np.array([])

        watched_by_user = set(self.watched.watched_dict[user_id])

        int_user_id = self.mappings.users_mapping[user_id]
        knn_recs = self.models.user_knn.similar_items(
            int_user_id,
            N=self.N_users
        )
        knn_recs.sort(key=lambda t: t[1], reverse=True)
        filtered_knn_recs: List[Tuple[int, float]] = []

        def filter_func(x: int) -> bool:
            return x not in watched_by_user

        def add_idf(_sim: float) -> Callable[[int], Tuple[int, float]]:
            return lambda x: (x, self.item_idf.item_idf_dict[x] * _sim)

        for user, sim in knn_recs:
            ext_user = self.mappings.users_inv_mapping[user]
            if ext_user == user_id:
                continue
            watched = list(filter(
                filter_func,
                self.watched.watched_dict[ext_user]
            ))
            watched_by_user.update(watched)
            filtered_knn_recs.extend(map(
                add_idf(sim),
                watched
            ))

        filtered_knn_recs.sort(key=lambda t: t[1], reverse=True)
        return np.array(list(map(lambda t: t[0], filtered_knn_recs[:k_recs])))

    def _predict_popular(self,
                         k_recs: int
                         ) -> np.ndarray:
        """
        Make popularity baseline recommendations.

        Parameters
        ----------
        k_recs : int
            Count of items to recommend

        Returns
        ----------
        np.ndarray
            Recommended items' IDs.
        """

        popularity_list = self.models.popular.popularity_list

        reco = popularity_list[0][:k_recs]

        return self.dataset.item_id_map.convert_to_external(reco)

    def predict(self, user_id: int, k_recs: int) -> List[int]:
        """
        Make recommendations for curtain user with userKNN and baseline.

        Parameters
        ----------
        user_id : int
            User ID
        k_recs : int
            Count of items to recommend

        Returns
        ----------
        List[int]
            Recommended items' IDs.
        """

        if not self.is_fitted:
            raise ValueError("Please call fit before predict")

        if user_id not in self.mappings.users_mapping:
            return self._predict_popular(k_recs).tolist()
        watched = self.watched.watched_dict[user_id]

        recs = self._predict_userknn(user_id, k_recs)

        popular = self._predict_popular(k_recs + len(watched))

        merged = np.concatenate((recs, popular))
        indexes = np.unique(
            merged,
            return_index=True
        )[1]

        return [merged[idx] for idx in sorted(indexes)][:k_recs]

    def _generate_recs_mapper(self,
                              model: ItemItemRecommender) \
            -> Callable[[int], Tuple[List[int], List[float]]]:
        def _recs_mapper(user: int) -> Tuple[List[int], List[float]]:
            user_id = self.mappings.users_mapping[user]
            recs = model.similar_items(user_id, N=self.N_users)
            return [self.mappings.users_inv_mapping[user]
                    for user, _ in recs], \
                   [sim for _, sim in recs]

        return _recs_mapper

    def recommend(self,
                  test: pd.DataFrame,
                  k_recs: int = 10) -> pd.DataFrame:
        """
        Make recommendations for batch of users with userKNN and baseline.

        Parameters
        ----------
        test : pd.DataFrame
            Users' IDs
        k_recs : int, optional, default ``10``
            Count of items to recommend

        Returns
        ----------
        pd.DataFrame
            Recommended items' IDs for users.
        """

        if not self.is_fitted:
            raise ValueError("Please call fit before predict")

        mapper = self._generate_recs_mapper(self.models.user_knn)

        recs = pd.DataFrame({'user_id': test['user_id']})
        recs['sim_user_id'], recs['sim'] = zip(*recs['user_id'].map(mapper))
        recs = recs.set_index('user_id').apply(pd.Series.explode).reset_index()

        recs = recs[~(recs['sim'] >= 0.999999)] \
            .merge(self.watched.watched,
                   left_on=['sim_user_id'],
                   right_on=['user_id'],
                   how='left')
        recs = recs \
            .explode('item_id') \
            .sort_values(['user_id', 'sim'], ascending=False) \
            .drop_duplicates(['user_id', 'item_id'], keep='first') \
            .merge(self.item_idf.item_idf, left_on='item_id', right_on='index',
                   how='left')

        recs['score'] = recs['sim'] * recs['idf']
        recs = recs.sort_values(['user_id', 'score'], ascending=False)

        popular = self.models.popular.recommend(
            test['user_id'].unique(),
            self.dataset,
            k_recs,
            True
        )
        recs = pd.concat([
            recs[['user_id', 'item_id', 'score']],
            popular[['user_id', 'item_id', 'score']],
        ], ignore_index=True).reset_index()

        recs['rank'] = recs.groupby('user_id').cumcount() + 1

        res = recs[recs['rank'] <= k_recs][
            ['user_id', 'item_id', 'score', 'rank']]

        return res
