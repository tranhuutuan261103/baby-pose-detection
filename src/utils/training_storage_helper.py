import json
import pickle
import os

def save_params(params: dict, file_path: str):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(params, f)

def save_model(model, file_name: str):
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, "wb") as file:
        pickle.dump(model, file)