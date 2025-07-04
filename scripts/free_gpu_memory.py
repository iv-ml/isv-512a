from gpuinfonv import GPUInfo

gpu_info = GPUInfo()

# Print information about all detected GPUs
gpu_info.print_gpu_info()

# Free up GPU resources
for i in range(8):
    gpu_info.free_up_gpu(i)  # Free up GPU 0

# num_gpus = torch.cuda.device_count()
# for gpu_id in range(num_gpus):
#     torch.cuda.set_device(gpu_id)
#     torch.cuda.empty_cache()
