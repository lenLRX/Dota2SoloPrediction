import sys
import os
from replay_parser import ReplayPool, ReplayDataSet
from model import SoloPredictionNet
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np


def collate_wrapper(batch):
    return batch


if __name__ == "__main__":


    cuda = torch.device('cuda')
    cpu = torch.device('cpu')

    data_set = ReplayDataSet(sys.argv[1])

    net = SoloPredictionNet().to(cuda)
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    writer = SummaryWriter()

    batch_size = 60
    iter_num = 0

    save_interval = 20

    while True:
        loader = DataLoader(data_set, batch_size=batch_size, num_workers=12, collate_fn=collate_wrapper)

        for batch_data in loader:
            for game in batch_data:
                game.to(cuda)
            optimizer.zero_grad()
            batch_loss = 0
            for game in batch_data:
                probs = net(game).view(-1)
                n_frames = len(probs)
                importance = np.arange(n_frames).astype("float32") / n_frames
                importance_tensor = torch.from_numpy(importance).to(cuda)

                loss = probs - game.rad_win
                loss *= importance_tensor
                loss = (loss * loss).mean()
                batch_loss += loss.item()
                loss.backward()
            batch_loss /= batch_size
            print("batch loss {}".format(batch_loss))
            writer.add_scalar("loss", batch_loss, iter_num)
            optimizer.step()
            #for game in batch_data:
            #    game.to(cpu)  # save memory

            iter_num += 1
            if iter_num % save_interval == 0:
                with open("save/SoloPred_{}.model".format(iter_num), "wb") as f:
                    torch.save(net.state_dict(), f)

