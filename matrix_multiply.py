import numpy as np
import matmul as mm

mm.init()

A = np.random.randint(0,100,(10,5))
B = np.random.randint(0,100,(5,10))

C = mm.matrix_multiply(A,B)


print(C - np.matmul(A,B))


mm.finalize()