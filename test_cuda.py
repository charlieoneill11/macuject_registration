import torch

if torch.cuda.is_available():
    print("CUDA is available on this system!")
else:
    print("CUDA is not available on this system.")