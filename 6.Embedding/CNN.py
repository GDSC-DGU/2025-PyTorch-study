# CNN
import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Linear(32 * 32 * 32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x)
        x = self.fc(x)
        return x
# 모델 생성
model = CNN()

# CNN 모델 아키텍처 출력
print("CNN 모델 아키텍처:")
print(model)

# 각 레이어별 파라미터 정보 출력
print("\n각 레이어별 파라미터 정보:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")