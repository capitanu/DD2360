#!/usr/bin/env python

import matplotlib.pyplot as plt
import os

iters = list(range(1,6))
times = []
tpbs = []
first = 16
while(first <= 512):
    tpbs.append(first)
    first = first * 2
for tpb in tpbs:
    for i in iters:
        result = os.popen("./bonus_exercise {} {}".format(i, tpb)).read()
        times.append(float(result))
    plt.plot(iters, times, label = "TPB = {}".format(tpb))
    times = []

plt.xlabel("Number of iteration per thread")
plt.ylabel("Seconds")
plt.legend()
plt.title("Time graph for increasing number of iterations with 100000000 threads")
plt.savefig("plot2.png")
plt.show()
