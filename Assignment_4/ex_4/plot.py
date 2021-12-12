import matplotlib.pyplot as plt
import numpy as np




array_size = [1000, 10000, 100000, 1000000, 10000000]
cpu = [0.000001, 0.000012, 0.000120, 0.001210, 0.012105]
gpu_opencl = [0.000016, 0.000015, 0.000020, 0.000254, 0.000195]
gpu_openacc = [0.072125, 0.101177, 0.095127, 0.094514, 0.107011]
gpu_cuda = [0.000022, 0.000032, 0.000216, 0.001300, 0.012343]

plt.plot(array_size, cpu, label="CPU time")
plt.plot(array_size, gpu_opencl, label="OpenCL time")
plt.plot(array_size, gpu_openacc, label="OpenACC time")
plt.plot(array_size, gpu_cuda, label="CUDA time")
plt.legend()
plt.ylabel("Time (ms)")
plt.xlabel("Array size")
plt.savefig("plot3.png")
plt.show()
