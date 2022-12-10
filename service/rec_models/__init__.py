from os.path import exists
from typing import Dict

from .base_model import BaseRecModel
from .test_model import TestModel
from .userknn import UserKnn

modelByName: Dict[str, BaseRecModel] = {
    "test_model": TestModel(),
    "userknn_with_popularity":
        UserKnn.load("userknn.dill")
        if exists("dumps/userknn.dill")
        else None,
}
