import random

import gym
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image


# <class 'gym.wrappers.time_limit.TimeLimit'>
# unwrapped : <class 'gym.envs.classic_control.cartpole.CartPoleEnv'>
# env = gym.make('CartPole-v1', render_mode="human").unwrapped
env = gym.make('CartPole-v0', render_mode="rgb_array")


# CartPole-v1일때만 적용되는 내용인데
resize = T.Compose(
    [T.ToPILImage(), T.Resize(40, interpolation=Image.CUBIC), T.ToTensor()]
)


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)


def get_screen():
    screen = env.render().transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(
            cart_location - view_width // 2,
            cart_location + view_width // 2
        )
    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # 크기를 수정하고 배치 차원(BCHW)을 추가하십시오.
    return resize(screen).unsqueeze(0)
