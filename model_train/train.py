# model_train/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_data_loaders
from model import CNNModel
from test import test
import mlflow
from torchinfo import summary

def train(model, device, loss_fn, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
        train_acc = correct / total
        
        if batch_idx % 100 == 0:
            step = epoch * len(train_loader) + batch_idx
            mlflow.log_metric('train_loss', loss.item(), step=step)
            mlflow.log_metric('train_accuracy', train_acc, step=step)
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

if __name__ == "__main__":
    mlflow.set_experiment("model-train")
    
    with mlflow.start_run():
        # 모델 설정
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        epochs = 10
        loss_fn = nn.NLLLoss()
        model = CNNModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 데이터 로더 가져오기
        train_loader, test_loader = get_data_loaders()
        
        # 매개변수 로깅
        params = {
            'epochs': epochs,
            'learning_rate': 0.001,
            'train_batch_size': 64,
            'test_batch_size': 1000,
            'loss_function': loss_fn.__class__.__name__,
            'optimizer': 'Adam'
        }
        mlflow.log_params(params)
        
        # 모델 요약 정보 로깅
        with open('model_summary.txt', 'w') as f:
            f.write(str(summary(model, input_size=(64, 1, 28, 28))))
        mlflow.log_artifact('model_summary.txt')
        
        best_test_loss = float('inf')
        
        # 모델 학습 및 평가
        for epoch in range(1, epochs + 1):
            train(model, device, loss_fn, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, device, loss_fn, test_loader)
            
        # 테스트 성능 로깅
        mlflow.log_metric('test_loss', test_loss, step=epoch)
        mlflow.log_metric('test_accuracy', test_acc, step=epoch)
            
        # 최적 모델 저장
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), "../models/best.pth")
            mlflow.pytorch.log_model(model, "best_model")
        torch.save(model.state_dict(), "../models/last.pth")
        
        # 최종 모델 저장
        mlflow.pytorch.log_model(model, "final_model")
