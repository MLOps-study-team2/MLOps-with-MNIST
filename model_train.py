# poetry shell
# python model_train.py
# 학습 후
# 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
from model import CNNModel

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    return train_loss / len(train_loader)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({accuracy:.2f}%)\n')
    return test_loss, accuracy

def main():
    mlflow.set_tracking_uri("http://localhost:5000")
    # experiment를 active하고 experiment instance를 반환.
    mlflow.set_experiment("MNIST_CNN_Classification")
    
    # 새로운 run을 시작. 기존 run이 있다면 end_run으로 끝내고 start_run을 해야 함.
    with mlflow.start_run():
        # MNIST dataset loading
        mnist = fetch_openml('mnist_784', version=1)
        X, y = mnist["data"], mnist["target"].astype(int)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/7, random_state=42)

        # Scaling (0-1 range)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Convert y_train and y_test to numpy arrays
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, 28, 28)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, 28, 28)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        # Create DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

        # Set up the device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        model = CNNModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Log model parameters
        mlflow.log_param("optimizer", type(optimizer).__name__)
        mlflow.log_param("learning_rate", optimizer.param_groups[0]['lr'])
        mlflow.log_param("epochs", 10)

        # Model training and evaluation
        for epoch in range(1, 11):  # 10 epochs
            train_loss = train(model, device, train_loader, optimizer, epoch)
            test_loss, accuracy = test(model, device, test_loader)
            
            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("test_loss", test_loss, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)

        # Log the final model
        # mlflow registry에 모델 저장
        # 원래 mlflow.sklearn이었는데 생각해보니 내 모델은 sklearn이 아님.
        # 그래서 나중에 다시 해봤더니 안되면 sklearn으로 다시 바꿔서 해보기
        mlflow.pytorch.log_model(model, "model", registered_model_name='MLOps-MNIST')
        #np.save("./ckpt/stat.npy", np.array([scaler.mean_, scaler.scale_]))
        #torch.save(model.state_dict(), './ckpt/model.pt')
        # active_run : 현재 active인 run을 object를 반환
        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

if __name__ == "__main__":
    main()