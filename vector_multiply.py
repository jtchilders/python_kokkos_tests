import numpy as np
import vecmul as vm
import time

vm.init()

array_size = 1000000000
a = np.random.randint(0,100,array_size)
b = np.random.randint(0,100,array_size)

print('tests using integer arrays of ', array_size, ' elements')
# openmp test timing
start = time.time()
c = vm.vector_multiply_int(a,b)
end = time.time()
print( f'{"openmp: ":<10} {end - start:.4f}')

# numpy test timing
start = time.time()
c = np.multiply(a,b)
end = time.time()
print( f'{"numpy: ":<10} {end - start:.4f}')

# serial test timing
start = time.time()
c = vm.vector_multiply_serial_int(a,b)
end = time.time()
print( f'{"serial: ":<10} {end - start:.4f}')



a = np.random.rand(array_size)
b = np.random.rand(array_size)

print('tests using float arrays of ', array_size, ' elements')

# openmp test timing
start = time.time()
c = vm.vector_multiply_float(a,b)
end = time.time()
print( f'{"openmp: ":<10} {end - start:.4f}')

# numpy test timing
start = time.time()
c = np.multiply(a,b)
end = time.time()
print( f'{"numpy: ":<10} {end - start:.4f}')

# serial test timing
start = time.time()
c = vm.vector_multiply_serial_float(a,b)
end = time.time()
print( f'{"serial: ":<10} {end - start:.4f}')



vm.finalize()