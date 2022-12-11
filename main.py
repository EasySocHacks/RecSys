import os

import numpy as np
import pandas as pd
import uvicorn
from rectools import Columns


from service.api.app import create_app
from service.rec_models.factoring_machine_model import FactoringMachineModel, \
    FMHyperParams, FMTuneParams, FMFeatures
from service.settings import get_config

config = get_config()
app = create_app(config)

ROOT_DIR = os.path.abspath(os.curdir)


if __name__ == "__main__":
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    #
    # interactions = pd.read_csv(f'{ROOT_DIR}/kion_train/interactions.csv')
    # users = pd.read_csv(f'{ROOT_DIR}/kion_train/users.csv')
    # items = pd.read_csv(f'{ROOT_DIR}/kion_train/items.csv')
    #
    # cold_users = list(set(users["user_id"]) - set(interactions["user_id"]))
    # cold_users.append(0)
    # data = {
    #     "user_id": cold_users,
    #     "item_id": [0] * len(cold_users),
    #     "last_watch_dt": "1990-01-01",
    #     "total_dur": [0] * len(cold_users),
    #     "watched_pct": [0] * len(cold_users)
    # }
    # new_interactions = pd.DataFrame(data)
    # interactions = interactions.append(new_interactions)
    #
    # cold_items = list(set(items["item_id"]) - set(interactions["item_id"]))
    # cold_items.append(0)
    # data = {
    #     "user_id": [0] * len(cold_items),
    #     "item_id": cold_items,
    #     "last_watch_dt": "1990-01-01",
    #     "total_dur": [0] * len(cold_items),
    #     "watched_pct": [0] * len(cold_items)
    # }
    # new_interactions = pd.DataFrame(data)
    # interactions = interactions.append(new_interactions)
    #
    # interactions[Columns.Weight] = \
    #     np.where(interactions['watched_pct'] > 10, 3, 1)
    # interactions[Columns.Weight].value_counts(normalize=True)
    #
    # Columns.Datetime = 'last_watch_dt'
    #
    # interactions[Columns.Datetime] = pd.to_datetime(
    #     interactions[Columns.Datetime], format='%Y-%m-%d')
    #
    # user_features_frames = []
    # for feature in ["sex", "age", "income"]:
    #     feature_frame = users.reindex(columns=[Columns.User, feature])
    #     feature_frame.columns = ["id", "value"]
    #     feature_frame["feature"] = feature
    #     user_features_frames.append(feature_frame)
    # user_features = pd.concat(user_features_frames)
    #
    # items["genre"] = items["genres"].str.lower() \
    #     .str.replace(", ", ",", regex=False).str.split(",")
    # genre_feature = items[["item_id", "genre"]].explode("genre")
    # genre_feature.columns = ["id", "value"]
    # genre_feature["feature"] = "genre"
    #
    # content_feature = items.reindex(columns=[Columns.Item, "content_type"])
    # content_feature.columns = ["id", "value"]
    # content_feature["feature"] = "content_type"
    #
    # item_features = pd.concat((genre_feature, content_feature))
    #
    # FactoringMachineModel(
    #     interactions,
    #     FMHyperParams(
    #         64,
    #         "warp",
    #         0.01,
    #         0,
    #         0,
    #     ),
    #     FMTuneParams(
    #         1,
    #         16,
    #         123,
    #     ),
    #     FMFeatures(
    #         user_features,
    #         ["sex", "age", "income"],
    #         item_features,
    #         ["genre", "content_type"],
    #     ),
    #     True
    # )

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8080"))

    uvicorn.run(app, host=host, port=port)
