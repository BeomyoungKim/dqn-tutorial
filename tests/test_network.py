import pytest
import torch

from src.models.network import DQN


def test_network():
    # CartPole-v1은 action 개수가 2개
    network = DQN((40, 90), 2)
    # Conv2d는 (batch_size, channel_size, height, width)
    output = network(torch.rand((1, 3, 40, 90)))
    assert output.size() == torch.Size([1, 2])
