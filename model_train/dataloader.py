# model_train/dataloader.py
from torchvision import datasets
from torch.utils.data import DataLoader
from transform import get_mnist_transform

def get_data_loaders(root='data/', batch_size=128, num_workers=2):
    mnist_transform = get_mnist_transform()
    
    # MNIST 데이터셋 로드 (훈련 및 테스트 세트)
    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=mnist_transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=mnist_transform)
    
    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader
