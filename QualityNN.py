import torch


class QualityNN(torch.nn.Module):
    def __init__(self, observation_space, action_space):
        super(QualityNN, self).__init__()
        self.layer1 = torch.nn.Linear(observation_space, 16)
        self.layer2 = torch.nn.Linear(16, 64)
        self.layer3 = torch.nn.Linear(64, action_space)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.layer2(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.layer3(x)

        # simply use the output value (expected return)
        return x