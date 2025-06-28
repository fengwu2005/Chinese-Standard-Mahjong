import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels//16, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels//16, out_channels, 1),
            nn.Sigmoid()
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        se_weight = self.se(out)
        out = out * se_weight
        
        out += residual
        return torch.relu(out)

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        tower_layers = [
            nn.Conv2d(143, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        ]
        num_res_blocks = 8 
        for _ in range(num_res_blocks):
            tower_layers.append(ResidualBlock(256, 256))

        tower_layers += [
            ResidualBlock(256, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 64),
            ResidualBlock(64, 32),
            nn.Flatten()
        ]
        self._tower = nn.Sequential(*tower_layers)

        # 保持与原版相同的输出维度
        self._logits = nn.Sequential(
            nn.Linear(32 * 4 * 9, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512, 235)
        )
        
        self._value_branch = nn.Sequential(
            nn.Linear(32 * 4 * 9, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        # 更精细的参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_dict):
        obs = input_dict["observation"].float()
        hidden = self._tower(obs)
        logits = self._logits(hidden)
        mask = input_dict["action_mask"].float()
        inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
        masked_logits = logits + inf_mask
        value = self._value_branch(hidden)
        return masked_logits, value