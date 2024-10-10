import json
import pickle

def save_params(params, file_path: str):
    with open(file_path, 'w') as f:
        json.dump(params, f)

def save_model(model, file_name):
    with open(file_name, "wb") as file:
        pickle.dump(model, file)