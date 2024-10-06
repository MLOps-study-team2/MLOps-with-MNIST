'''
BentoML

이미지 인덱스를 입력으로 제공하면, 서비스는 해당 이미지에 대한 예측 결과를 반환합니다.
'''
import torch
import bentoml
from bentoml.io import NumpyNdarray, JSON
from sklearn.datasets import fetch_openml



# GPU, MPS 또는 CPU 사용 여부 확인
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("device: ", device)


# BentoML에 저장된 모델 로드
runner = bentoml.pytorch.get('saved_model_bentoml').to_runner()

# 서비스 정의
svc = bentoml.Service(name='mnist_classifier', runners=[runner])

# API 엔드포인트 정의
@svc.api(input=NumpyNdarray(dtype="int32", shape=(1,)), output=JSON())

async def predict(idx):
    try:
        # MNIST 데이터셋 로드
        mnist = fetch_openml('mnist_784', version=1)
        X, y = mnist["data"], mnist["target"].astype(int)

        # 입력 데이터 준비
        input_data = X.iloc[idx].values
        input_data = torch.tensor(input_data, dtype=torch.float32).view(-1, 1, 28, 28)
        input_data = input_data.to(device)
        gt_label = y.iloc[idx].item()

        # 모델 예측
        pred = await runner.async_run(input_data)
        pred_label = pred.argmax(dim=1).item()

        # 결과
        result = {
            "idx": idx,
            "input_shape": input_data.shape,
            "gt_label": gt_label,
            "pred_label": pred_label

        }
        return result

    except IndexError:
        return {"error": "Invalid index. Please provide a valid index within the MNIST dataset range."}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"} 
