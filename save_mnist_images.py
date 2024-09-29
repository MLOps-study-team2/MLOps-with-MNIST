import torch
from torchvision import datasets
import torchvision.transforms as transforms
from PIL import Image
import os

# 저장할 디렉토리 생성
save_dir = 'mnist_samples'
os.makedirs(save_dir, exist_ok=True)

# MNIST 테스트 데이터셋 로드
transform = transforms.Compose([
    transforms.ToTensor()
])
test_dataset = datasets.MNIST(root='data/', train=False, download=True, transform=transform)

# 이미지 저장
for i in range(10):  # 10개의 이미지를 저장합니다.
    img_tensor, label = test_dataset[i]
    img_array = img_tensor.numpy()[0] * 255  # 텐서를 이미지로 변환
    img = Image.fromarray(img_array.astype('uint8'), mode='L')
    img.save(f'{save_dir}/mnist_{label}_{i}.png')

print(f"이미지가 '{save_dir}' 디렉토리에 저장되었습니다.")