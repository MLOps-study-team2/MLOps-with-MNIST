from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn.functional as F
import mlflow.pytorch
import numpy as np
import io

# 모델 불러오기
model = mlflow.pytorch.load_model("mlruns/132498781976046875/b2d1080be20f4d08a47840fded44aae5/artifacts/cnn_model")  # <your_run_id>을 실행 ID로 대체
model.eval()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# FastAPI 앱 생성
app = FastAPI()

# 이미지 전처리 함수 (MNIST 이미지 크기인 28x28로 변환)
def preprocess_image(image: Image.Image):
    # 이미지를 흑백으로 변환하고 크기 조정
    image = image.convert("L").resize((28, 28))
    image_np = np.array(image) / 255.0  # 픽셀 값을 [0, 1]로 스케일링
    tensor = torch.tensor(image_np, dtype=torch.float32).view(1, 1, 28, 28)
    return tensor

# 추론 엔드포인트 (이미지 파일을 입력으로 받음)
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 이미지 파일을 열어서 PIL 이미지로 변환
        image = Image.open(io.BytesIO(await file.read()))
        tensor = preprocess_image(image)
        
        tensor = tensor.to(device)
        
        # 모델 추론 수행
        with torch.no_grad():
            output = model(tensor)
            prediction = torch.argmax(F.softmax(output, dim=1), dim=1).item()

        return JSONResponse(content={"prediction": prediction})

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")

# 서버 상태 확인을 위한 기본 엔드포인트
@app.get("/")
def read_root():
    return {"message": "CNN model inference API is up and running"}
