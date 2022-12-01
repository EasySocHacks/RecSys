from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple, Union

import dill
import numpy as np
import pandas as pd
import scipy as sp
from implicit.nearest_neighbours import ItemItemRecommender
from rectools.dataset import Dataset
from rectools.models.popular import PopularModel

from .base_model import BaseRecModel


class Mappings:
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


class UserKnn(BaseRecModel):
    """Class for fit-predict UserKNN model
       based on ItemKNN model from implicit.nearest_neighbours
    """

    def __init__(self, model: ItemItemRecommender) -> None:
        self.user_knn: ItemItemRecommender = model
        self.is_fitted: bool = False

        self.mappings: Optional[Mappings] = None

        self.watched: Optional[pd.DataFrame] = None

        self.popular: PopularModel = PopularModel()

        self.item_idf: Optional[pd.DataFrame] = None
        self.dataset: Optional[Dataset] = None

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
                   user_col: str = 'user_id',
                   item_col: str = 'item_id',
                   weight_col: str = None) \
            -> sp.sparse.coo_matrix:
        if weight_col:
            weights = df[weight_col].astype(np.float32)
        else:
            weights = np.ones(len(df), dtype=np.float32)

        interaction_matrix = sp.sparse.coo_matrix((
            weights,
            (
                df[user_col].map(self.mappings.users_mapping.get),
                df[item_col].map(self.mappings.items_mapping.get)
            )
        ))

        self.watched = df.groupby(user_col).agg({item_col: list})
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
        self.item_idf = item_idf

    def fit(self, train: pd.DataFrame,
            save: bool = False) -> None:
        self.get_mappings(train)
        weights_matrix = self.get_matrix(train)

        self._count_item_idf(train, train.shape[0])

        self.user_knn.fit(weights_matrix)

        self.dataset = Dataset.construct(
            train,
        )
        self.popular.fit(self.dataset)

        self.is_fitted = True

        if save:
            with open('dumps/userknn.dill', 'wb') as f:
                dill.dump(self, f)

    def _generate_recs_mapper(self,
                              model: ItemItemRecommender,
                              N_users: int = 50) \
            -> Callable[[int], Tuple[List[int], List[float]]]:
        def _recs_mapper(user: int) -> Tuple[List[int], List[float]]:
            user_id = self.mappings.users_mapping[user]
            recs = model.similar_items(user_id, N=N_users)
            return [self.mappings.users_inv_mapping[user]
                    for user, _ in recs], \
                   [sim for _, sim in recs]

        return _recs_mapper

    @staticmethod
    def momentum(args: Union[pd.Series, pd.DataFrame],
                 moment: float = 0.01) -> int:
        res = 0
        for x in reversed(args.index):
            res = x * (1 - moment) + moment * res
        return res

    def split_cold_users(self, users: pd.DataFrame) \
            -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        unq_users = pd.DataFrame({'user_id': users['user_id'].unique()})
        unq_users['user_internal_id'] = \
            unq_users['user_id'].map(self.mappings.users_mapping)
        return unq_users, \
            unq_users[unq_users['user_internal_id'].notnull()], \
            unq_users[~unq_users['user_internal_id'].notnull()]

    def _predict_userknn(self, user_id: int) -> pd.DataFrame:
        if self.mappings.users_mapping[user_id] is None:
            return pd.DataFrame([], columns=['user_id', 'score', 'model'])
        mapper = self._generate_recs_mapper(self.user_knn)

        sim_users, sims = mapper(user_id)
        recs = pd.DataFrame({'sim_user_id': sim_users, 'sim': sims})

        recs = recs[~(recs['sim'] >= 0.999999)] \
            .merge(self.watched,
                   left_on=['sim_user_id'],
                   right_on=['user_id'],
                   how='left') \
            .explode('item_id') \
            .sort_values(['sim'], ascending=False) \
            .drop_duplicates(['item_id'], keep='first') \
            .merge(self.item_idf, left_on='item_id', right_on='index',
                   how='left')

        recs['score'] = recs['sim'] * recs['idf']
        recs = recs.sort_values('score', ascending=False)
        recs['model'] = "userKNN"

        return recs

    def _predict_popular(self,
                         k_recs: int,
                         ) -> pd.DataFrame:
        popularity_list = self.popular.popularity_list

        reco = popularity_list[0][:k_recs]
        scores = popularity_list[1][:k_recs]

        popular = pd.DataFrame(
            {
                'item_id': self.dataset.item_id_map.convert_to_external(
                    reco
                ),
                'score': scores,
            }
        )

        popular['model'] = "popular"

        return popular

    def predict(self, user_id: int, k_recs: int) -> List[int]:
        if not self.is_fitted:
            raise ValueError("Please call fit before predict")

        if self.mappings.users_mapping[user_id] is None:
            return self._predict_popular(k_recs)["item_id"].to_list()
        watched = self.watched['item_id'][user_id]

        recs = self._predict_userknn(user_id)

        popular = self._predict_popular(2 * k_recs + len(watched))

        # print(recs.head())
        # print()
        # print(popular.head())
        # print()

        res = pd.concat([
            recs['item_id'],
            popular['item_id'],
        ], ignore_index=True)
        res = res[~res.isin(watched)].unique()[:k_recs].tolist()

        # print(res)
        # print()

        return res

    def recommend(self, test: pd.DataFrame, k_recs: int = 10) -> pd.DataFrame:
        _, users, _ = self.split_cold_users(test)

        if not self.is_fitted:
            raise ValueError("Please call fit before predict")

        mapper = self._generate_recs_mapper(self.user_knn)

        recs = pd.DataFrame({'user_id': users['user_id']})
        recs['sim_user_id'], recs['sim'] = zip(*recs['user_id'].map(mapper))
        recs = recs.set_index('user_id').apply(pd.Series.explode).reset_index()

        recs = recs[~(recs['sim'] >= 0.999999)] \
            .merge(self.watched, left_on=['sim_user_id'], right_on=['user_id'],
                   how='left') \
            .explode('item_id') \
            .sort_values(['user_id', 'sim'], ascending=False) \
            .drop_duplicates(['user_id', 'item_id'], keep='first') \
            .merge(self.item_idf, left_on='item_id', right_on='index',
                   how='left')
        #   .groupby(['user_id', 'sim']).agg(UserKnn.momentum) \

        recs['score'] = recs['sim'] * recs['idf']
        recs = recs.sort_values(['user_id', 'score'], ascending=False)
        recs['model'] = "userKNN"

        popular = self.popular.recommend(
            users['user_id'].to_numpy(),
            self.dataset,
            k_recs,
            True
        )
        popular['model'] = "popular"

        print(recs.head())
        print(popular.head())

        res = pd.concat([
            recs[['user_id', 'item_id', 'score', 'model']],
            popular[['user_id', 'item_id', 'score', 'model']],
        ], ignore_index=True).reset_index()

        print(res.head(10))

        recs['rank'] = recs.groupby('user_id').cumcount() + 1

        res = recs[recs['rank'] <= k_recs][
            ['user_id', 'item_id', 'score', 'rank', 'model']]

        return res
