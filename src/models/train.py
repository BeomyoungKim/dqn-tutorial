from itertools import count

import torch

from src.utils.image import plot_durations
from src.models.env import screen_cartpole_env as env
from src.config import device, TARGET_UPDATE, NUM_EPISODES


def train_cartpole(agent):
    # loss가 줄고 있는가
    # reward가 커지고 있는가
    episode_durations = []
    episode_losses = []
    for i in range(NUM_EPISODES):
        state, _ = env.reset()
        episode_loss = 0
        for t in count():
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            agent.memory.push(state, action, next_state, reward)
            state = next_state

            loss = agent.optimize()
            # episode_loss += loss['loss']

            if done:
                episode_durations.append(t + 1)
                episode_losses.append(episode_loss / t)
                plot_durations(episode_durations)
                # print(episode_losses)
                print(loss)
                break
        if i and i % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
    # what rendered?
    env.render()
    env.close()
