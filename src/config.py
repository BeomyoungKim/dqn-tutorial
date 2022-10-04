import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 5000
TARGET_UPDATE = 100
CAPACITY = 65536
NUM_EPISODES = 1000
