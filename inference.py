import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import os
from model import CNNModel

def de_norm(x, mean, std):
    zero_idxs = torch.where(x == 0.0)[0]
    x = mean + std * x
    x[zero_idxs] = 0.0
    return x

def load_model():
    model = CNNModel()
    state_dict = torch.load('./ckpt/model.pt')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_scaler():
    return torch.tensor(np.load(os.path.join('ckpt', "stat.npy")), dtype=torch.float)

def inference(model, x):
    model.eval()
    x = x.reshape(-1, 1, 28, 28)
    with torch.no_grad():
        output = model(x)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    return pred

if __name__ == "__main__":
    model = CNNModel()
    state_dict = torch.load('./ckpt/model.pt')
    model.load_state_dict(state_dict)
    mnist = fetch_openml('mnist_784', version=1)
    mean, std = torch.tensor(np.load(os.path.join('ckpt', "stat.npy")), dtype=torch.float)
    X, y = torch.from_numpy(mnist["data"].to_numpy()[0]), mnist["target"].astype(int).to_numpy()[0]
    X_test  = de_norm(X, mean, std)
    pred = inference(model, X_test).item()