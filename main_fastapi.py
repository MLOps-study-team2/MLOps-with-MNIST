'''
FastAPI
'''
import random
import mlflow
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


@app.post("/predict")
def predict(
    idx: int,
):    
    global model
    global device
    try:
        mnist = fetch_openml('mnist_784', version=1)
        X, y = mnist["data"], mnist["target"].astype(int)

        input_data = X.iloc[idx]
        input_data = torch.tensor(input_data, dtype=torch.float32).view(-1, 1, 28, 28)
        input_data = input_data.to(device)
        gt_label = y.iloc[idx].item()

        pred = model(input_data)
        pred_label = pred.argmax(dim=1).item()

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


    
@app.post("/v2/predict")
def predict(idx: int):
    try:
        # Set up MLflow
        mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Replace with your MLflow tracking server URI
        model_name = "mnist_model"
        model_version = "1"  # Replace with the desired version

        # Load the model from the registry
        loaded_model = mlflow.pytorch.load_model(f"models:/{model_name}/{model_version}")
        loaded_model.to(device)

        mnist = fetch_openml('mnist_784', version=1)
        X, y = mnist["data"], mnist["target"].astype(int)

        input_data = X.iloc[idx]
        input_data = torch.tensor(input_data, dtype=torch.float32).view(-1, 1, 28, 28)
        input_data = input_data.to(device)
        gt_label = y.iloc[idx].item()

        pred = loaded_model(input_data)
        pred_label = pred.argmax(dim=1).item()

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
