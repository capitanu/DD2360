import numpy as np
import matplotlib.pyplot as plt

n_groups = 3
tsize = (32, 32, 32)
bandwidth = (25.1 , 24.6, 578.9)


# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, tsize, bar_width,
alpha=opacity,
color='b',
label='Transfer Size (GB)')

rects2 = plt.bar(index + bar_width, bandwidth, bar_width,
alpha=opacity,
color='g',
label='Bandwidth (GB/s)')

plt.title('NVIDIA GeForce RTX 3080 Bandwidth Test')
plt.xticks(index + bar_width, ('Host to Devide', 'Device to Host', 'Device to Device'))
plt.legend()

plt.tight_layout()
plt.savefig("plt1.png")

