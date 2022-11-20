from typing import Dict

from .base_model import BaseRecModel as BaseRecModel
from .test_model import TestModel as TestModel


modelByName: Dict[str, BaseRecModel] = {
    "test_model": TestModel()
}
