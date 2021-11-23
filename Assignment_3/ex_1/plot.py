import matplotlib.pyplot as plt
import numpy as np

first = [13.879000, 25.234000, 8.386000]
second = [13.466000, 24.023000, 8.612000]
third = [14.918000, 27.380000, 9.684000]
x = [1, 2, 3]

n_bins = 10

photos = ['hk', 'nyc', 'rome']
fig, ax0 = plt.subplots()
ax0.hist(first, 3, density=True, histtype='bar', label=photos)
ax0.legend(prop={'size': 10})
ax0.set_title('Different images')
plt.show()
