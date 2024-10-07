from fastapi import FastAPI, File, UploadFile
from PIL import Image   # 이미지 처리
import io
import torch
import numpy as np
from inference import load_model, load_scaler, de_norm, inference
import uvicorn
import matplotlib.pyplot as plt
#from io import BytesIO
#import base64   # base64 인코딩

app = FastAPI() # FastAPI 인스턴스 생성

# 어플리케이션을 실행할 때나 종료하기 직전에 특정 함수를 호출
# startup : 어플리케이션 시작 직후 지정한 함수를 호출
# shutdown : 어플리케이션이 종료되기 직전에 지정한 함수를 호출
@app.on_event("startup")    
async def load_model_and_stats():   # 어플리케이션 시작할 때 이 메소드를 호출
    global model, mean, std
    model = load_model()
    mean, std = load_scaler()
    
'''def visualize_prediction(image_array, prediction):
    plt.figure(figsize=(4, 4))
    plt.imshow(image_array.reshape(28, 28), cmap='gray')
    plt.title(f'Predicted: {prediction}')
    plt.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf'''
    
# 서버에 데이터를 저장하고자 할 때 사용
# 학습한 모델의 inference 결과를 반환
@app.post("/predict/")
# UploadFile : 이미지, 문서 등을 업로드 받아야 할 때
# PredictIn : json 형식의 데이터 등 구조화된 데이터를 입력으로 받을 때
async def predict(file: UploadFile = File(...)):    
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image).astype(np.float32)
    image_tensor = torch.from_numpy(image_array)
    X_test = de_norm(image_tensor.flatten(), mean, std)
    pred = inference(model, X_test).item()
    #viz_buffer = visualize_prediction(image_array, pred)
    
    # Convert the visualization to base64
    #viz_base64 = base64.b64encode(viz_buffer.getvalue()).decode()
    
    return pred

if __name__ == "__main__":
    # 스크립트가 실행될 때 uvicorn 서버를 시작
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)