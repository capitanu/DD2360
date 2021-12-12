import matplotlib.pyplot as plt
import numpy as np




array_size = [1000, 10000, 100000, 1000000, 10000000]
cpu = [0.000001, 0.000012, 0.000120, 0.001210, 0.012105]
gpu_opencl = [0.000016, 0.000015, 0.000020, 0.000254, 0.000195]
gpu_openacc = [0.072125, 0.101177, 0.095127, 0.094514, 0.107011]

plt.plot(array_size, cpu, label="CPU time")
plt.plot(array_size, gpu_opencl, label="OpenCL time")
plt.plot(array_size, gpu_openacc, label="OpenACC time")
plt.legend()
plt.ylabel("Time (ms)")
plt.xlabel("Array size")
plt.savefig("plot2.png")
plt.show()
