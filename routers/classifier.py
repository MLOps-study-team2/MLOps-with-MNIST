# routers/classifier.py
from fastapi import APIRouter, Request, File, UploadFile
import io
from PIL import Image
from model_train.transform import get_mnist_transform
from utils.utils import get_class_name
from fastapi.responses import JSONResponse
import torch

router = APIRouter() # 라우터 생성

@router.get("/")
async def root():
    return {"message": "MNIST Digit Classifier API"}

@router.post("/predict/", response_class=JSONResponse)
async def img_predict(request: Request, file: UploadFile = File(...)):
    try:
        # 업로드된 파일 읽기
        contents = await file.read()
        # 이미지를 PIL 객체로 로드하고 그레이스케일로 변환
        pil_image = Image.open(io.BytesIO(contents)).convert('L')
        pil_image = pil_image.resize((28, 28))  # Resize to 28x28
        transform = get_mnist_transform() # 이미지 전처리
        torch_image = transform(pil_image).unsqueeze(0) # 배치 차원 추가
        
        model = request.app.state.model # 모델 가져오기
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_image = torch_image.to(DEVICE)
        model.to(DEVICE)
        
        # 모델 예측
        output = model(torch_image)
        predicted_class_name = get_class_name(output)
        return {"class_name": predicted_class_name}
    except Exception as e:
        # 예외 처리
        return JSONResponse(status_code=500, content={"error": str(e)})