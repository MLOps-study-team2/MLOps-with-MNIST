# utils/utils.py
import torch
import os
from model_train.model import CNNModel

def get_class_name(prediction):
    predicted_class_index = prediction.argmax(dim=1).item()
    return str(predicted_class_index) # 예측된 클래스 인덱스를 문자열로 반환

def load_model(root="models/", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(root, "best.pth")
    model = CNNModel().to(device) # 모델 인스턴스 생성
    model.load_state_dict(torch.load(model_path, map_location=device)) # 저장된 모델 가중치 로드
    model.eval()
    return model