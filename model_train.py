import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import mlflow
import bentoml


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    # Save
    # torch.save(model, './saved_model.pt')
    torch.save(model.state_dict(), './saved_model.pt')

    # Save for bentoml
    bentoml.pytorch.save_model('saved_model_bentoml', model, labels={'owner':'aitech', 'stage':'dev'})

    # Save mlflow
    mlflow.pytorch.log_model(model, 'model')

    # log
    # mlflow.log_metric('train loss', loss.item())


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()  # 배치 손실 더하기
            pred = output.argmax(dim=1, keepdim=True)  # 가장 높은 log-probability를 가진 인덱스 찾기
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({100. * correct / len(test_loader.dataset):.0f}%)\n')
          
    # log
    # mlflow.log_metric('test loss', test_loss)

def main():
    # MNIST 데이터셋 로드
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist["data"], mnist["target"].astype(int)

    # 데이터를 학습용과 테스트용으로 나누기
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, random_state=42)

    # 스케일링 (0~1 범위로)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # y_train과 y_test를 numpy 배열로 변환
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    
    # PyTorch 텐서로 변환
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, 28, 28)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, 28, 28)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # DataLoader 생성
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    # 모델 설정
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("device: ", device)

    model = CNNModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 모델 학습 및 평가
    for epoch in range(1, 11):  # 10 에포크 동안 학습
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    test(model, device, test_loader)

    # log
    # mlflow.log_param('epoch', 10)
    # mlflow.log_param('lr', 0.001)
    # mlflow.log_param('train batch size', 64)
    # mlflow.log_param('test batch size', 1000)
    
if __name__ == "__main__":
    print("Start Training...")
    
    # log
    mlflow.autolog()

    # Experimetn 설정
    mlflow.set_experiment('mnist')

    # with mlflow.start_run():
    
    main()
