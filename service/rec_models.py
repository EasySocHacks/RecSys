import typing as tp


class BaseRecModel:
    def recommend(self, user_id: int, k_recs: int) -> tp.List[int]:
        pass


class TestModel(BaseRecModel):
    def recommend(self, user_id: int, k_recs: int) -> tp.List[int]:
        return list(range(k_recs))


modelByName = {
    "test_model": TestModel()
}
