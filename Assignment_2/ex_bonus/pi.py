#!/usr/bin/env python

import matplotlib.pyplot as plt
import os
import numpy as np
import math

iters = 100
result = os.popen("./bonus_exercise {} {}".format(iters, 128)).read()
result = result.split(" ")
result = result[:-1]
result = np.array(result)
result = result.astype(np.float)
iters = list(range(1, 101))
pi = []
for i in iters:
    pi.append(math.pi)

plt.plot(iters, pi, label="Reference PI")
plt.plot(iters, result, label="Resulting PI")
plt.legend()
plt.xlabel("Iterations")
plt.savefig("plot3.png")
plt.show()

