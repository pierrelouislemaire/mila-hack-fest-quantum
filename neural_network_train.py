import matplotlib.pyplot as plt
import numpy as np 
import time
import torch
import torch.nn as nn
import torch.optim as optim

def set_up_gpus():
    if torch.cuda.is_available():
        print(f"GPUs: {torch.cuda.get_device_name(0)} is available.")

if __name__ == "__main__":
    timer = time.time()

    # 1. Set GPUs usage
