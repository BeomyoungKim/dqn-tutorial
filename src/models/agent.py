import math
import random

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import (
    device,
    BATCH_SIZE,
    GAMMA,
    EPS_END,
    EPS_START,
    EPS_END,
    EPS_DECAY,
    CAPACITY
)
from src.models.network import DQN
from src.utils.buffer import Transition, ReplayMemory


class Agent:
    def __init__(self, height, width, n_actions):
        # 2차원 input에 대해 일반화된 agent
        input_dim = (height, width)
        self.n_actions = n_actions

        self.policy_net = DQN(input_dim, n_actions).to(device)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.criterion = nn.SmoothL1Loss()

        self.target_net = DQN(input_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(CAPACITY)

        self.steps_done = 0

    @torch.no_grad()
    def select_action(self, state):
        sample = random.random()
        eps_threshold = (
            EPS_END +
            (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        )
        self.steps_done += 1
        if sample > eps_threshold:
            # torch.max(input=tensor, dim=1) return (max, max_indices)
            return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # n_actions 가짓수를 가진 (1, 1) tensor
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

    def optimize(self):
        if len(self.memory) < BATCH_SIZE:
            return {'loss' : 0}
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device, dtype=torch.bool
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # torch.gather(input=tensor, dim=1, index=tensor)
        # return out[i][j][k] = input[i][index[i][j][k]][k] if dim == 1
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        # detach : no gradient differentiate tensor copy
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # 기대 Q 값 계산
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # 모델 최적화
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return {'loss': loss.item()}
