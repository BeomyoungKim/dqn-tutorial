from itertools import count

import torch

from src.models.env import env, get_screen
from src.utils.image import plot_durations
from src.config import device, TARGET_UPDATE, NUM_EPISODES


def run(agent):
    # loss가 줄고 있는가
    # reward가 커지고 있는가
    episode_durations = []
    episode_losses = []
    for i in range(NUM_EPISODES):
        env.reset()
        current_screen = last_screen = get_screen()
        state = current_screen - last_screen
        episode_loss = 0
        for t in count():
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            last_screen = current_screen
            current_screen = get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

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
    env.render()
    env.close()
