import numpy as np
import vecmul as vm
import time

vm.init()
vm.print_hw_config()

array_size = 2**19
a = np.random.randint(0,100,array_size)
b = np.random.randint(0,100,array_size)

print('tests using integer arrays of ', array_size, ' elements')
# openmp test timing
start = time.time()
c = vm.vector_multiply_int(a,b)
end = time.time()
kokkos_time = end - start
print( f'{"on-device: ":<30} {end - start:.4g}  ({(end - start) / kokkos_time:.4g})')

# test timing with vectorization
start = time.time()
c = vm.vector_multiply_vv_int(a,b)
end = time.time()
print( f'{"vectorized on-device: ":<30} {end - start:.4g}  ({ (end - start) / kokkos_time:.4g})')

# numpy test timing
start = time.time()
c = np.multiply(a,b)
end = time.time()
print( f'{"numpy: ":<30} {end - start:.4g}  ({ (end - start) / kokkos_time:.4g})')

# serial test timing
start = time.time()
c = vm.vector_multiply_serial_int(a,b)
end = time.time()
print( f'{"serial: ":<30} {end - start:.4g}  ({ (end - start) / kokkos_time:.4g})')



a = np.random.rand(array_size)
b = np.random.rand(array_size)

print('tests using float arrays of ', array_size, ' elements')

# openmp test timing
start = time.time()
c = vm.vector_multiply_float(a,b)
end = time.time()
kokkos_time = end - start
print( f'{"on-device: ":<30} {end - start:.4g}  ({ (end - start) / kokkos_time:.4g})')

# test timing with vectorization
start = time.time()
c = vm.vector_multiply_vv_float(a,b)
end = time.time()
print( f'{"vectorized on-device: ":<30} {end - start:.4g}  ({ (end - start) / kokkos_time:.4g})')

# numpy test timing
start = time.time()
c = np.multiply(a,b)
end = time.time()
print( f'{"numpy: ":<30} {end - start:.4g}  ({ (end - start) / kokkos_time:.4g})')

# serial test timing
start = time.time()
c = vm.vector_multiply_serial_float(a,b)
end = time.time()
print( f'{"serial: ":<30} {end - start:.4g}  ({ (end - start) / kokkos_time:.4g})')




vm.finalize()