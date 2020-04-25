import numpy as np
import os
import random
import re
from itertools import cycle

import torch
from torch.utils.data import Dataset

from GameState_pb2 import GameState

CREEP_DIM = 21
HERO_DIM = 25


class ReplayDataSet(Dataset):
    def __init__(self, base_dir):
        super(ReplayDataSet).__init__()
        files = os.listdir(base_dir)
        self.file_list = [os.path.join(base_dir, file) for file in files if file.endswith(".protobin")]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        return ReplayBatch(SingleReplay(self.file_list[item]), torch.device("cpu"))


class ReplayBatch(object):

    def __init__(self, replay, device):
        self.device = device
        frames = replay.frames
        hero_batch = 1
        creep_batch = 20

        self.rad_hero_state = []
        self.dire_hero_state = []
        self.rad_creep_state = []
        self.dire_creep_state = []
        for i, frame in enumerate(frames):
            rad_hero_batch = np.zeros((hero_batch, HERO_DIM), dtype="float32")
            dire_hero_batch = np.zeros((hero_batch, HERO_DIM), dtype="float32")
            rad_creep_batch = np.zeros((creep_batch, CREEP_DIM), dtype="float32")
            dire_creep_batch = np.zeros((creep_batch, CREEP_DIM), dtype="float32")

            rad_hero_count = 0
            dire_hero_count = 0
            rad_creep_count = 0
            dire_creep_count = 0
            for h_state in frame.hero_states:
                if h_state[17] > 0 and rad_hero_count < hero_batch:
                    rad_hero_batch[rad_hero_count] = h_state
                    rad_hero_count += 1
                elif h_state[18] > 0 and dire_hero_count < hero_batch:
                    dire_hero_batch[dire_hero_count] = h_state
                    dire_hero_count += 1

            random.shuffle(frame.creep_states)

            for c_state in frame.creep_states:
                if c_state[17] > 0 and rad_creep_count < creep_batch:
                    rad_creep_batch[rad_creep_count] = c_state
                    rad_creep_count += 1
                elif c_state[18] > 0 and dire_creep_count < creep_batch:
                    dire_creep_batch[dire_creep_count] = c_state
                    dire_creep_batch += 1

            self.rad_hero_state.append(rad_hero_batch)
            self.dire_hero_state.append(dire_hero_batch)
            self.rad_creep_state.append(rad_creep_batch)
            self.dire_creep_state.append(dire_creep_batch)

        self.rad_hero_state = torch.from_numpy(np.asarray(self.rad_hero_state)).unsqueeze(0).to(device)
        self.dire_hero_state = torch.from_numpy(np.asarray(self.dire_hero_state)).unsqueeze(0).to(device)
        self.rad_creep_state = torch.from_numpy(np.asarray(self.rad_creep_state)).unsqueeze(0).to(device)
        self.dire_creep_state = torch.from_numpy(np.asarray(self.dire_creep_state)).unsqueeze(0).to(device)
        self.rad_win = int(replay.radiant_win)

    def features(self):
        return self.rad_hero_state, self.dire_hero_state, self.rad_creep_state, self.dire_creep_state

    def to(self, device):
        self.rad_hero_state = self.rad_hero_state.to(device)
        self.dire_hero_state = self.dire_hero_state.to(device)
        self.rad_creep_state = self.rad_creep_state.to(device)
        self.dire_creep_state = self.dire_creep_state.to(device)

    @staticmethod
    def merge_batch(replays):
        pass


class ReplayPool(object):

    def __init__(self, base_dir):
        self.base_dir = base_dir
        files = os.listdir(self.base_dir)
        self.replays = []
        for file in files:
            if file.endswith(".protobin"):
                print("loading {}".format(file))
                self.replays.append(SingleReplay(os.path.join(self.base_dir, file)))
                if len(self.replays) > 2:
                    break

    def get_batch(self, device):
        batches = [ReplayBatch(rep, device) for rep in self.replays]
        return batches


class SingleReplay(object):
    def __init__(self, replay_path):
        self.replay_path = replay_path
        with open(replay_path, "rb") as f:
            self.replay_proto = GameState()
            self.replay_proto.ParseFromString(f.read())

        self.build_buffer()

    def build_buffer(self):
        self.radiant_win = self.replay_proto.radiant_win
        self.frames = []

        for i, frame in enumerate(self.replay_proto.frames):
            if i % 8 != 0:
                continue
            self.frames.append(Frame(frame, self.radiant_win))


class Frame(object):

    def __init__(self, proto_frame, label):
        self.proto_frame = proto_frame
        self.label = label
        self.creep_states = []
        self.hero_states = []
        for creep_state in self.proto_frame.creep_states:
            np_creep_state = np.zeros((21,), dtype="float32")
            np_creep_state[0] = creep_state.m_cellX
            np_creep_state[1] = creep_state.m_cellY
            np_creep_state[2] = creep_state.m_cellZ
            np_creep_state[3] = creep_state.m_vecX
            np_creep_state[4] = creep_state.m_vecY
            np_creep_state[5] = creep_state.m_vecZ
            np_creep_state[6] = creep_state.m_lifeState
            np_creep_state[7] = creep_state.m_iDamageMin
            np_creep_state[8] = creep_state.m_iDamageMax
            np_creep_state[9] = creep_state.m_iDamageBonus
            np_creep_state[10] = creep_state.m_flMaxMana
            np_creep_state[11] = creep_state.m_iGoldBountyMin
            np_creep_state[12] = creep_state.m_iGoldBountyMax
            np_creep_state[13] = creep_state.m_flHealthRegen
            np_creep_state[14] = creep_state.m_iMaxHealth
            np_creep_state[15] = creep_state.m_iHealth
            np_creep_state[16] = creep_state.m_iXPBounty
            np_creep_state[17] = creep_state.m_bTeamRadiant
            np_creep_state[18] = creep_state.m_bTeamDire
            np_creep_state[19] = creep_state.m_bVisibleByRadiant
            np_creep_state[20] = creep_state.m_bVisibleByDire
            self.creep_states.append(np_creep_state)

        for hero_state in self.proto_frame.hero_states:
            np_hero_state = np.zeros((25,), dtype="float32")
            np_hero_state[0] = hero_state.m_cellX
            np_hero_state[1] = hero_state.m_cellY
            np_hero_state[2] = hero_state.m_cellZ
            np_hero_state[3] = hero_state.m_vecX
            np_hero_state[4] = hero_state.m_vecY
            np_hero_state[5] = hero_state.m_vecZ
            np_hero_state[6] = hero_state.m_lifeState
            np_hero_state[7] = hero_state.m_iDamageMin
            np_hero_state[8] = hero_state.m_iDamageMax
            np_hero_state[9] = hero_state.m_iDamageBonus
            np_hero_state[10] = hero_state.m_flMaxMana
            np_hero_state[11] = hero_state.m_iGoldBountyMin
            np_hero_state[12] = hero_state.m_iGoldBountyMax
            np_hero_state[13] = hero_state.m_flHealthRegen
            np_hero_state[14] = hero_state.m_iMaxHealth
            np_hero_state[15] = hero_state.m_iHealth
            np_hero_state[16] = hero_state.m_iXPBounty
            np_hero_state[17] = hero_state.m_bTeamRadiant
            np_hero_state[18] = hero_state.m_bTeamDire
            np_hero_state[19] = hero_state.m_bVisibleByRadiant
            np_hero_state[20] = hero_state.m_bVisibleByDire
            np_hero_state[21] = hero_state.m_iCurrentLevel
            np_hero_state[22] = hero_state.m_flStrengthTotal
            np_hero_state[23] = hero_state.m_flAgilityTotal
            np_hero_state[24] = hero_state.m_flIntellectTotal
            self.hero_states.append(np_hero_state)