import json


def check_gradient_magnitudes(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def load_json(loc):
    with open(loc, "r") as f:
        data_dic = json.load(f)
    return data_dic


def save_json(save_loc: str, data):
    with open(save_loc, "w") as file:
        json.dump(data, file, indent=1)
