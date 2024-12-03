import torch

# Check number of available GPUs
num_gpus = torch.cuda.device_count()
print(f"Available GPUs: {num_gpus}")

# Choose the GPU by ID, e.g., using GPU 1
gpu_id = 1
torch.cuda.set_device(gpu_id)
device = torch.device(f"cuda:{gpu_id}")
print(f"Using device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(device)}")
