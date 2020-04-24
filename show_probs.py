import sys
import torch
from model import SoloPredictionNet

from replay_parser import ReplayPool
import matplotlib.pyplot as plot


if __name__ == "__main__":
    pool = ReplayPool(sys.argv[1])
    batch = pool.get_batch(torch.device("cpu"))
    rep = batch[0]

    m = torch.load("save/SoloPred_2660.model")
    net = SoloPredictionNet()
    net.load_state_dict(m)

    preds = net(rep).view(-1).detach().numpy().tolist()
    print("radiant win {}".format(rep.rad_win))
    plot.plot(preds)
    plot.title("radiant win {}".format(rep.rad_win))
    plot.ylabel("rad win prob")
    plot.xlabel("time (tick)")
    plot.show()


