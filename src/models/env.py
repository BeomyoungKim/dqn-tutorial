import random

import gym
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image


# <class 'gym.wrappers.time_limit.TimeLimit'>
# unwrapped : <class 'gym.envs.classic_control.cartpole.CartPoleEnv'>
# env = gym.make('CartPole-v1', render_mode="human").unwrapped
# env = gym.make('CartPole-v0', render_mode="rgb_array")


class ScreenCartPole(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.resize = T.Compose(
            [T.ToPILImage(), T.Resize(40, interpolation=Image.CUBIC), T.ToTensor()]
        )
        self.last_screen = None

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        if not terminated:
            current_screen = self.get_screen()
            observation = current_screen - self.last_screen
            self.last_scrren = current_screen
        else:
            observation = None

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        current_screen = last_screen = self.get_screen()
        self.last_screen = last_screen

        _, _, screen_height, screen_width = last_screen.shape
        n_actions = self.env.action_space.n
        info = {
            'screen_height': screen_height,
            'screen_width' : screen_width,
            'n_actions' : n_actions
        }

        return current_screen - last_screen, info

    def _get_cart_location(self, screen_width):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)

    def get_screen(self):
        screen = self.env.render().transpose((2, 0, 1))
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = self._get_cart_location(screen_width)
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
        return self.resize(screen).unsqueeze(0)


env = gym.make('CartPole-v0', render_mode="rgb_array")
screen_cartpole_env = ScreenCartPole(env)
