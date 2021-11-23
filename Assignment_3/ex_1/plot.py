import matplotlib.pyplot as plt
import numpy as np

hk = [12.528000, 13.111000, 15.059000]
nyc = [23.151000, 22.851000, 27.076000]
rome = [8.533000, 8.765000, 9.524000]
x = [1, 2, 3]
x_ticks = ["B/W", "Gaussian", "Edges"]

n_bins = 10

photos = ['hk', 'nyc', 'rome']
plt.xticks(x, x_ticks)
plt.plot(x, hk, label = "hk")
plt.plot(x, nyc, label = "nyc")
plt.plot(x, rome, label = "rome")
plt.legend()
plt.ylabel("Time (ms)")
plt.savefig("plot1.png")
plt.show()
