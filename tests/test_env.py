import torch
import pytest
import matplotlib.pyplot as plt

from src.models.env import screen_cartpole_env as env


def test_env():
    """wrapped cartpole environment by screen state
    """
    state, _ = env.reset()
    # 기존 CartPole의 state 4개의 features (4,)의 np.ndarray
    # (1, 3, 40, 90)의 Tensor로 변경
    assert type(state) == torch.Tensor
    assert state.shape == (1, 3, 40, 90)
    # action 개수 2개
    assert env.action_space.n == 2
    random_action = env.action_space.sample()
    # step 후 state shape이 동일
    observation, reward, terminated, truncated, info = env.step(random_action)
    assert terminated in [True, False]
    assert reward in [0, 1]
    assert type(observation) == torch.Tensor
    assert observation.shape == (1, 3, 40, 90)
    env.close()


def test_state():
    """screen으로 wrap한 state이 올바르게 출력되는 지 확인
    """
    env.reset()
    plt.figure()
    plt.imshow(
        env.get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
        interpolation='none'
    )
    plt.title('Example extracted screen')
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    env.close()
