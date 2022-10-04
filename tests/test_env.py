import pytest
import numpy as np
import matplotlib.pyplot as plt

from src.models.env import env, get_screen


def test_env():
    state, info = env.reset()
    # state 4개의 features
    assert type(state) == np.ndarray
    assert state.shape == (4,)
    # action 개수 2개
    assert env.action_space.n == 2
    random_action = env.action_space.sample()
    # step 후 state shape이 동일
    observation, reward, terminated, truncated, info = env.step(random_action)
    print(type(observation))
    assert terminated in [True, False]
    assert reward in [0, 1]
    assert type(observation) == np.ndarray
    assert observation.shape == (4,)
    env.close()


def test_state():
    env.reset()
    plt.figure()
    plt.imshow(
        get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
        interpolation='none'
    )
    plt.title('Example extracted screen')
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    env.close()
