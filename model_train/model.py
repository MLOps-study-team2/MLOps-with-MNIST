# model_train/model.py
import torch.nn as nn
import torch.nn.functional as F  # 활성화 함수 등 사용을 위해 임포트

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # 합성곱 계층 정의
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 풀링 계층 정의
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 완전 연결 계층 정의
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 순전파 정의
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # 텐서를 펼쳐서 입력으로 사용
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # 출력 계층
        return F.log_softmax(x, dim=1) # 클래스 확률 출력
