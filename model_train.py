import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
from tqdm import tqdm


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

def train(model:nn.Module, device, train_loader:DataLoader, optimizer:optim.Adam, epoch):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output:torch.Tensor = model(data)
        loss:torch.Tensor = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
                    
    return total_loss / len(train_loader)

def test(model:nn.Module, device, test_loader:DataLoader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output:torch.Tensor = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()  # 배치 손실 더하기
            pred = output.argmax(dim=1, keepdim=True)  # 가장 높은 log-probability를 가진 인덱스 찾기
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f"loss: {test_loss} accuracy: {accuracy}")
    
    return test_loss, accuracy

def main():
    # MNIST 데이터셋 로드
    mnist:pd.DataFrame = fetch_openml('mnist_784', version=1)
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
    
    model = CNNModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # MLflow 시작
    with mlflow.start_run() as run:
        # 파라미터 로깅
        mlflow.log_param("learning rate", .001)
        mlflow.log_param("batch_size", 64)
        
        # 모델 학습 및 평가
        for epoch in tqdm(range(1, 11), desc="training..."):
            train_loss = train(model, device, train_loader, optimizer, epoch)
            test_loss, accuracy = test(model, device, test_loader)
            
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("test_loss", test_loss, step=epoch)
            mlflow.log_metric("test_accuracy", accuracy, step=epoch)
        
        input_example = X_test_tensor[0:1].to(device)
        input_example_in_numpy = input_example.cpu().numpy()
        output_example = model(input_example).detach().cpu().numpy()
        signature = infer_signature(input_example.cpu().numpy(), output_example)
        
        mlflow.pytorch.log_model(model, "cnn_model", signature=signature, input_example=input_example_in_numpy)
        
        print(f"Run ID: {run.info.run_id}")
    
if __name__ == "__main__":
    mlflow.set_experiment("mnist_experiment")
    main()