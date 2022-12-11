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

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8080"))

    uvicorn.run(app, host=host, port=port)
