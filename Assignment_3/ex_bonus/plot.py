import matplotlib.pyplot as plt
import numpy as np

x = np.arange(7)
cpu = [0.751000, 5.815000, 47.443000, 368.392000, 2968.284000, 23683.770000, 191485.124000]
cublas = [0.205000, 0.270000, 0.310000, 0.223000, 0.400000, 0.998000, 6.396000]
gpu_global = [0.015000, 0.021000, 0.086000, 0.612000, 4.928000, 38.110000, 23.464000]
gpu_shared = [0.009000, 0.010000, 0.023000, 0.118000, 0.830000, 7.281000, 52.357000]
width = 0.1
x_ticks = ["64" , "128", "256", "512", "1024", "2048", "4096"]
plt.xticks(x, x_ticks)
plt.bar(x-0.2, cpu, width, label= "CPU")
#plt.bar(x-0.1, cublas, width, label = "GPU cuBLAS")
#plt.bar(x, gpu_global, width, label = "GPU Global")
#plt.bar(x+0.1, gpu_shared, width, label = "GPU Shared")
plt.ylabel("Time (ms)")
plt.legend()
plt.savefig("gpu3.png")
plt.show()
