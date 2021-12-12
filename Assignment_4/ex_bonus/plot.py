import matplotlib.pyplot as plt
import numpy as np




array_size = [1000, 10000, 100000]
cpu = [0.061030, 0.608230, 6.115252]
gpu_16 = [0.000843, 0.003809, 0.053141]
gpu_32 = [0.000848, 0.004856, 0.066645]
gpu_64 = [0.000897, 0.003559, 0.065236]
gpu_128 = [0.000943, 0.004303, 0.065573]
gpu_256 = [0.004488, 0.008764, 0.075314]

plt.plot(array_size, gpu_16, label="GPU 16 block size -> time")
plt.plot(array_size, gpu_32, label="GPU 32 block size -> time")
plt.plot(array_size, gpu_64, label="GPU 64 block size -> time")
plt.plot(array_size, gpu_128, label="GPU 128 block size -> time")
plt.plot(array_size, gpu_256, label="GPU 256 block size -> time")

plt.legend()
plt.ylabel("Time (ms)")
plt.xlabel("NUM_PARTICLES")
plt.savefig("plot5.png")
plt.show()
