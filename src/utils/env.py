import torch

def get_device(device_arg: str = "auto") -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return "cuda"

    return "cuda" if torch.cuda.is_available() else "cpu"