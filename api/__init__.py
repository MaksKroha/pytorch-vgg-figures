from fastapi import FastAPI
from src.model.neural_net import CNN

import torch
app = FastAPI()
dropout = 0.2
trained_model_path = "models/trained_model.pt"
model = CNN(dropout)
# TODO: env file reading for hyperparams


@app.on_event("startup")
async def startup_event():
    model.load_state_dict(torch.load(trained_model_path))


from .endpoints.grid import get_canvas
from .endpoints.process import process_pixels