from typing import Dict

from .base_model import BaseRecModel
from .test_model import TestModel

modelByName: Dict[str, BaseRecModel] = {
    "test_model": TestModel()
}
