import dill
import pandas as pd
import numpy as np
import scipy as sp
from rectools.dataset import Dataset
from rectools.models.popular import PopularModel
from typing import Dict, Any, Tuple, Callable, Union, List, Iterable, \
    Reversible
from collections import Counter
from implicit.nearest_neighbours import ItemItemRecommender

from .base_model import BaseRecModel


class UserKnn(BaseRecModel):
    """Class for fit-predict UserKNN model
       based on ItemKNN model from implicit.nearest_neighbours
    """

    def __init__(self, model: ItemItemRecommender, N_users: int = 50) -> None:
        self.model = model
        self.is_fitted = False
        self.N_users = N_users

        self.users_inv_mapping = None
        self.users_mapping = None
        self.items_inv_mapping = None
        self.items_mapping = None

        self.watched = None

        self.user_knn = None
        self.weights_matrix = None
        self.n = None

        self.popular = PopularModel()

    def get_mappings(self, train) -> None:
        self.users_inv_mapping = dict(enumerate(train['user_id'].unique()))
        self.users_mapping = {v: k for k, v in self.users_inv_mapping.items()}

        self.items_inv_mapping = dict(enumerate(train['item_id'].unique()))
        self.items_mapping = {v: k for k, v in self.items_inv_mapping.items()}

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
                df[user_col].map(self.users_mapping.get),
                df[item_col].map(self.items_mapping.get)
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

    def _count_item_idf(self, df: pd.DataFrame) -> None:
        item_cnt = Counter(df['item_id'].values)
        item_idf = pd.DataFrame.from_dict(item_cnt, orient='index',
                                          columns=['doc_freq']).reset_index()
        item_idf['idf'] = item_idf['doc_freq'].apply(
            lambda x: self.idf(self.n, x))
        self.item_idf = item_idf

    def fit(self, train: pd.DataFrame,
            save: bool = False) -> None:
        self.user_knn = self.model
        self.get_mappings(train)
        self.weights_matrix = self.get_matrix(train)

        self.n = train.shape[0]
        self._count_item_idf(train)

        self.user_knn.fit(self.weights_matrix)

        self.dataset = Dataset.construct(
            train,
        )
        self.popular.fit(self.dataset)

        self.is_fitted = True

        if save:
            with open('dumps/userknn.dill', 'wb') as f:
                dill.dump(self, f)

    @staticmethod
    def _generate_recs_mapper(model: ItemItemRecommender,
                              user_mapping: Dict[int, int],
                              user_inv_mapping: Dict[int, int], N: int) \
        -> Callable[[int], Tuple[List[int], List[int]]]:
        def _recs_mapper(user):
            user_id = user_mapping[user]
            recs = model.similar_items(user_id, N=N)
            return [user_inv_mapping[user] for user, _ in recs], \
                   [sim for _, sim in recs]

        return _recs_mapper

    @staticmethod
    def momentum(args: Union[pd.Series, pd.DataFrame],
                 moment: int = 0.01) -> int:
        res = 0
        for x in reversed(args.index):
            res = x * (1 - moment) + moment * res
        return res

    def split_cold_users(self, users: pd.DataFrame) \
            -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        unq_users = pd.DataFrame({'user_id': users['user_id'].unique()})
        unq_users['user_internal_id'] = \
            unq_users['user_id'].map(self.users_mapping)
        return unq_users, \
            unq_users[unq_users['user_internal_id'].notnull()], \
            unq_users[~unq_users['user_internal_id'].notnull()]

    def predict(self, test: pd.DataFrame, k_recs: int = 10) -> pd.DataFrame:

        all_users, users, cold_users = self.split_cold_users(test)

        if not self.is_fitted:
            raise ValueError("Please call fit before predict")

        mapper = self._generate_recs_mapper(
            model=self.user_knn,
            user_mapping=self.users_mapping,
            user_inv_mapping=self.users_inv_mapping,
            N=self.N_users
        )

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
            self.dataset.user_id_map.external_ids,
            self.dataset,
            k_recs,
            False
        )
        popular['model'] = "popular"
        popular['user_id'].map(self.users_inv_mapping)

        cold_recs = self.dataset.interactions.df \
            .groupby('item_id') \
            .agg(score=('user_id', 'nunique')) \
            .sort_values(['score'], ascending=False) \
            .assign(key=1) \
            .head(k_recs) \
            .reset_index() \
            .merge(all_users.assign(key=1), on='key') \
            .drop("key", axis=1)
        cold_recs['model'] = 'common_popular'

        print(recs.head())
        print(popular.head())
        print(cold_recs)

        res = pd.concat([
            recs[['user_id', 'item_id', 'score', 'model']],
            popular[['user_id', 'item_id', 'score', 'model']],
            cold_recs[['user_id', 'item_id', 'score', 'model']],
        ], ignore_index=True)

        print(res.head(10))

        cold_recs = self.dataset.interactions.df \
            .groupby('item_id')['user_id'] \
            .agg("nunique") \
            .sort_values(ascending=False)

        recs['rank'] = recs.groupby('user_id').cumcount() + 1

        res = recs[recs['rank'] <= k_recs][
            ['user_id', 'item_id', 'score', 'rank', 'model']]

        return res
