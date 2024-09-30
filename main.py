'''
FastAPI
'''
import random
import torch
import torch.nn as nn

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Optional, Literal
from sklearn.datasets import fetch_openml

from model_train import CNNModel


import warnings
warnings.filterwarnings('ignore')


app = FastAPI()

model = None
@app.on_event("startup")
async def load_model():
    global model
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("device: ", device)

    model_path = './saved_model.pt'
    model = CNNModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()



@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/predict")
def predict(
    input_data: Optional[Any] = None,
):    
    global model
    global device
    
    if input_data == None:    # using scikit-learn data 
        mnist = fetch_openml('mnist_784', version=1)
        X, y = mnist["data"], mnist["target"].astype(int)

        idx = random.randint(0, len(X))
        input_data = X.iloc[idx]
        input_data = torch.tensor(input_data, dtype=torch.float32).view(-1, 1, 28, 28)
        input_data = input_data.to(device)
        gt_label = y.iloc[idx].item()

        pred = model(input_data)
        pred_label = pred.argmax(dim=1).item()

        result = {
            "input_shape": input_data.shape,
            "gt_label": gt_label,
            "pred_label": pred_label

        }
        return result

    else:
        pass

    




'''
BentoML
'''




# if __name__ == "__main__":

#     input_data = None

#     result = predict(input_data)
#     print("Result: ", result)
