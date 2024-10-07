import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import io
from PIL import Image   # 이미지 처리
import torch
import os
from inference import de_norm, inference

# mlflow tracking은 parameter, code version, metrics, output 등을 logging하는 것
# http://localhost:5000에 tracking을 저장한다.
mlflow.set_tracking_uri("http://localhost:5000")
model_name = 'MLOps-MNIST'
model_version = 2
# registry에 저장된 모델을 불러온다.
# 원래 mlflow.sklearn이었는데 생각해보니 내 모델은 sklearn이 아님.
# 그래서 나중에 다시 해봤더니 안되면 sklearn으로 다시 바꿔서 해보기
model = mlflow.pytorch.load_model(
    model_uri=f'models:/{model_name}/{model_version}'
)
# test할 사진 불러오기
file_path = './5.png'
mean, std = torch.tensor(np.load(os.path.join('ckpt', "stat.npy")), dtype=torch.float)
with open(file_path, 'rb') as file:
    image_data = file.read()
image = Image.open(io.BytesIO(image_data)).convert('L')  # Convert to grayscale
image = image.resize((28, 28))  # Resize to 28x28
image_array = np.array(image).astype(np.float32)
image_tensor = torch.from_numpy(image_array)
X_test = de_norm(image_tensor.flatten(), mean, std)
pred = inference(model, X_test).item()
print(pred)