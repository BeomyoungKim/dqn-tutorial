import torch.nn as nn
import torch.nn.functional as F

from src.config import device


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        # python 2.x
        # 상속받는 class의 __init__() 실행
        # super(DQN, self).__init__()
        super().__init__()
        # 2차원 input에 대해 일반화된 network
        height, width = input_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(16, 32, 5, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 5, 2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        conv_height_size = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        conv_width_size = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        linear_input_size = conv_height_size * conv_width_size * 32
        self.head = nn.Linear(in_features=linear_input_size, out_features=output_dim)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # view : only contiguous tensor
        # reshape : if non-contiguous "copy" (== .contiguous().view())
        return self.head(x.view(x.size(0), -1))

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file, device):
        self.load_state_dict(torch.load(file, map_location=device))
