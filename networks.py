import torch
from torch import nn

class NN(nn.Module):
    def __init__(self, in_channels, feature_size, num_residual, board_size, value_hidden_size):
        super().__init__()
        self.feature_size = feature_size
        self.num_residual = num_residual
        self.init_transform = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = feature_size, kernel_size = 3, padding=1),
            nn.BatchNorm2d(feature_size),
            nn.ReLU()
        )
        self.conv2d_f_f = nn.Conv2d(in_channels = feature_size, out_channels = feature_size, kernel_size = 3, padding=1)
        self.batchNorm = nn.BatchNorm2d(feature_size)
        self.relu = nn.ReLU()

        self.policy_conv2d = nn.Conv2d(in_channels = feature_size, out_channels = 2, kernel_size = 1, padding=0)
        self.batchNormPolicy = nn.BatchNorm2d(2)
        self.policy_full = nn.Linear( 2* board_size[0] * board_size[1], board_size[0] * board_size[1])

        self.value_conv2d = nn.Conv2d(in_channels = feature_size, out_channels = 1, kernel_size = 1, padding=0)
        self.batchNormValue = nn.BatchNorm2d(1)
        self.value_full_1 = nn.Linear(board_size[0] * board_size[1], value_hidden_size)
        self.value_full_2 = nn.Linear(value_hidden_size, 1)
        self.value_tanh = nn.Tanh()

        self.board_size = tuple(board_size)

    def policy_head(self, x):
        x = self.policy_conv2d(x)
        x = self.batchNormPolicy(x)
        x = self.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.policy_full(x)
        x = torch.reshape(x, (-1,)+self.board_size)
        return x

    def value_head(self, x):
        x = self.value_conv2d(x)
        x = self.batchNormValue(x)
        x = self.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.value_full_1(x)
        x = self.relu(x)
        x = self.value_full_2(x)
        x = self.value_tanh(x)
        return x
    
    def feature_transform(self, x):
        x = self.init_transform(x)
        for i in range(self.num_residual):
            y = self.conv2d_f_f(x)
            y = self.batchNorm(y)
            y = self.relu(y)
            y = self.conv2d_f_f(y)
            y = self.batchNorm(y)
            y = self.relu(y)
            y = self.conv2d_f_f(y)
            y = self.batchNorm(y)
            y = x + y
            x = self.relu(y)
        return x

    def forward(self, x):
        features = self.feature_transform(x)
        policy_logit = self.policy_head(features)
        value = self.value_head(features)
        return policy_logit, value

        
    
class ConvNN(nn.Module):
    def __init__(self, in_channels, feature_size, num_layer, value_hidden_size):
        super().__init__()
        self.feature_size = feature_size
        self.num_layer = num_layer
        self.init_transform = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = feature_size, kernel_size = 3, padding=1),
            nn.BatchNorm2d(feature_size),
            nn.ReLU()
        )
        self.conv2d_f_f = nn.Conv2d(in_channels = feature_size, out_channels = feature_size, kernel_size = 3, padding=1)
        self.batchNorm = nn.BatchNorm2d(feature_size)
        self.relu = nn.ReLU()

        self.policy_conv2d = nn.Conv2d(in_channels = feature_size, out_channels = 1, kernel_size = 1, padding=0)

        self.value_conv2d = nn.Conv2d(in_channels = feature_size, out_channels = feature_size, kernel_size = 1, padding=0)
        self.value_full_1 = nn.Linear(feature_size, value_hidden_size)
        self.value_full_2 = nn.Linear(value_hidden_size, 1)
        self.value_tanh = nn.Tanh()

    def policy_head(self, x):
        x = self.policy_conv2d(x)
        return x.squeeze(1)

    def value_head(self, x):
        x = self.value_conv2d(x)
        x = torch.mean(x, dim=(2,3))
        x = self.relu(x)
        x = self.value_full_1(x)
        x = self.relu(x)
        x = self.value_full_2(x)
        x = self.value_tanh(x)
        return x
    
    def feature_transform(self, x):
        x = self.init_transform(x)
        for _ in range(self.num_layer):
            y = self.conv2d_f_f(x)
            y = self.batchNorm(y)
            y = self.relu(y)
        return x

    def forward(self, x):
        features = self.feature_transform(x)
        policy_logit = self.policy_head(features)
        value = self.value_head(features)
        return policy_logit, value

        