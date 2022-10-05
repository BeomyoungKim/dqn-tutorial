import random
import typing
from collections import deque # namedtuple

import torch


# Transition = namedtuple(
#     'Transition',
#     ('state', 'action', 'next_state', 'reward')
# )
Transition = typing.NamedTuple(
    'Transition',
    [
        ('state', torch.Tensor),
        ('action', torch.Tensor),
        ('next_state', torch.Tensor),
        ('reward', torch.Tensor)
    ]
)


class ReplayMemory:
    # action에 의한 transition을 저장
    # 무작위로 샘플링
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """transition 무작위 샘플링"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
