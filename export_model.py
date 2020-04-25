import os
import torch

from model import SoloPredictionNet
from util import get_latest_params


if __name__ == "__main__":
    path = "./save"
    latest = get_latest_params(path)
    model = SoloPredictionNet()
    model.load_state_dict(torch.load(os.path.join(path, latest)))

    script = torch.jit.script(model)
    script.save("solo_predict.pt")

