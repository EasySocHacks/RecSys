from typing import Dict

from .base_model import BaseRecModel
from .test_model import TestModel
from .factoring_machine_model import FactoringMachineModel

modelByName: Dict[str, BaseRecModel] = {
    "test_model": TestModel(),
    "factoring_machine": FactoringMachineModel.load()
}
