import torch
import torch.nn as nn
import torch.nn.functional as F


class SoloPredictionNet(nn.Module):

    def __init__(self):
        super(SoloPredictionNet, self).__init__()
        self.hero_dim = 25
        self.creep_dim = 21
        self.hidden = 64
        self.lstm_hidden = 512
        self.rad_hero_fc = nn.Linear(self.hero_dim, self.hidden)
        self.dire_hero_fc = nn.Linear(self.hero_dim, self.hidden)
        self.rad_creep_fc = nn.Linear(self.creep_dim, self.hidden)
        self.dire_creep_fc = nn.Linear(self.creep_dim, self.hidden)
        self.lstm = nn.LSTM(self.hidden * 4, self.lstm_hidden, batch_first=True)
        self.prob_fc = nn.Linear(self.lstm_hidden, 1)

    def forward(self, rad_hero_state, dire_hero_state, rad_creep_state, dire_creep_state):
        o_rad_hero_state = F.relu(self.rad_hero_fc(rad_hero_state)).max(-2).values
        o_dire_hero_state = F.relu(self.dire_hero_fc(dire_hero_state)).max(-2).values
        o_rad_creep_state = F.relu(self.rad_creep_fc(rad_creep_state)).max(-2).values
        o_dire_creep_state = F.relu(self.dire_creep_fc(dire_creep_state)).max(-2).values

        hidden_state = torch.cat([o_rad_hero_state,
                   o_dire_hero_state,
                   o_rad_creep_state,
                   o_dire_creep_state], dim=-1)

        lstm_out, _ = self.lstm(hidden_state)
        probs = F.sigmoid(self.prob_fc(lstm_out))
        return probs






