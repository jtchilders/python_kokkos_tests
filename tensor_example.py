import torch
import tenex
import ctypes

tenex.init()
tenex.print_hw_config()

# Create a tensor on a device (e.g., CUDA)
tensor = torch.randn(10, device='cuda')  # Example tensor
print('starting tensor: ', tensor)

# Pass the data pointer of the tensor to the C++ extension
result = tenex.process_tensor(tensor.data_ptr(), tensor.size(0))

# Wrap the result pointer in a PyTorch tensor
# something like this
# output = torch.Tensor(result.data_ptr,result.size)

tenex.finalize()