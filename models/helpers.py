import torch

def load_saved_model(path, device):
    backbone, fc = torch.load(path, map_location=device)
    nclasses = fc.out_features if fc.out_features > 2 else 1
    model = BaseModel(backbone, fc, nclasses)
    model.to(device)
    return model
