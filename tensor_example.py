import torch
import tenex
import ctypes
import torch.utils.dlpack as thdlpack

tenex.init()
tenex.print_hw_config()


def wrap_in_tensor2(data_ptr, size):
   # Convert uintptr_t to a proper device pointer for PyTorch
   buffer = (ctypes.c_float * size).from_address(data_ptr)
   storage = torch.cuda.FloatStorage.from_buffer(buffer, size)
   tensor = torch.FloatTensor(storage).view(size)
   return tensor


def wrap_in_tensor(data_ptr, size):
   
   tensor = thdlpack.from_dlpack(data_ptr)
   return tensor

# Create a tensor on a device (e.g., CUDA)
tensor = torch.randn(10, device='cuda')  # Example tensor
print('starting tensor: ', tensor)

# Pass the data pointer of the tensor to the C++ extension
result = tenex.process_tensor(tensor.data_ptr(), tensor.size(0))

# Wrap the result pointer in a PyTorch tensor
output_tensor = wrap_in_tensor(result.data_ptr, result.size)
print('result tensor: ', output_tensor)


tenex.finalize()