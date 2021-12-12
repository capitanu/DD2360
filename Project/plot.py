import matplotlib.pyplot as plt
import numpy as np


x400 = [0.001578, 0.000804, 0.000414, 0.000221, 0.000128, 0.000133, 0.000140, 0.000135] 
x800 = [0.006497, 0.003138, 0.001590, 0.000815, 0.000424, 0.000431, 0.000436, 0.000433]
x1600 = [0.026288, 0.013008, 0.006785, 0.003426, 0.001774, 0.001621, 0.001627, 0.001622]

x = [1, 2, 3, 4, 5, 6, 7, 8]
x_ticks = [ "2", "4", "8" , "16", "32", "64", "128", "256"]

plt.xticks(x, x_ticks)
plt.plot(x, x400, label="400x400 Image Size")
plt.plot(x, x800, label="800x800 Image Size")
plt.plot(x, x1600, label="1600x1600 Image Size")
plt.legend()
plt.ylabel("Time (ms)")
plt.xlabel("Threads per block")
plt.savefig("plot1.png")
plt.show()


GPU = [0.000129, 0.000428, 0.001619, 0.006593]
CPU = [0.016652, 0.071616, 0.266327, 1.062895]

x = [1, 2, 3, 4]
x_ticks = [ "400x400", "800x800", "1600x1600" , "3200x3200"]

plt.xticks(x, x_ticks)
plt.plot(x, GPU, label="GPU (32 TPB)")
plt.plot(x, CPU, label="CPU")
plt.legend()
plt.ylabel("Time (ms)")
plt.xlabel("Image Size")
plt.savefig("plot2.png")
plt.show()


GPU = [0.001615, 0.001856, 0.002155, 0.002560, 0.003112, 0.003804]
CPU = [0.265073, 0.326033, 0.397898, 0.455299, 0.591881, 0.678103]

x = [1, 2, 3, 4, 5, 6]
x_ticks = [ "1.0", "1.1", "1.2" , "1.3", "1.4", "1.5"]

plt.xticks(x, x_ticks)
plt.plot(x, GPU, label="GPU (32 TPB)")
plt.plot(x, CPU, label="CPU")
plt.legend()
plt.ylabel("Time (ms)")
plt.xlabel("Sphere Radius")
plt.savefig("plot3.png")
plt.show()
