from numba import cuda
print(cuda.select_device(0))
print(cuda.select_device(1))
print(cuda.select_device(2))
print(cuda.select_device(3))
