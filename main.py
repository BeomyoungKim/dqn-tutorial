from src.models.agent import DQNAgent
from src.models.train import train_cartpole
from src.models.env import screen_cartpole_env


if __name__ == "__main__":
    _, info = screen_cartpole_env.reset()
    agent = DQNAgent(info['screen_height'], info['screen_width'], info['n_actions'])
    # various example agent for screen cartpole 
    # agent = DDQNAgent(info['screen_height'], info['screen_width'], info['n_actions'])
    train_cartpole(agent)
