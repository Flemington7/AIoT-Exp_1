import torch.nn as nn
import torch.nn.functional as F


class Weather_2NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 16)  # feature number is 13
        self.fc2 = nn.Linear(16, 1)

    def forward(self, inputs):
        tensor = F.sigmoid(self.fc1(inputs))
        tensor = self.fc2(tensor)
        return tensor
