from typing import Dict

from .base_model import BaseRecModel
from .factoring_machine_model import FactoringMachineModel
from .test_model import TestModel

modelByName: Dict[str, BaseRecModel] = {
    "test_model": TestModel(),
    "factoring_machine": FactoringMachineModel.load()
}
