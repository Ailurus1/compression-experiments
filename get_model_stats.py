import torch

def get_model_size_gb(model: torch.nn.Module) -> float:
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_all_gb = (param_size + buffer_size) / (1024**3)
    return size_all_gb

def get_model_params(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters())
