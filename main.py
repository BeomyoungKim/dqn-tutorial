from src.models.agent import Agent
from src.models.env import env, get_screen
from src.models.train import run as train


if __name__ == "__main__":
    env.reset()
    init_screen = get_screen()
    _, _, screen_height, screen_width = init_screen.shape
    n_actions = env.action_space.n

    agent = Agent(screen_height, screen_width, n_actions)
    train(agent)
