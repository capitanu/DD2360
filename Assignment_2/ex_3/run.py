#!/usr/bin/env python

import os
import matplotlib.pyplot as plt

iterations = 2000
particles = 1000
tpb = 32

while(tpb <= 512):
    gpu = []
    cpu = []
    iterations_list = []
    particles_list = []
    while(particles <= 100000):
        while(iterations <= 2000):
            result = os.popen("./exercise_3 {} {} {}".format(particles, iterations, tpb)).read()
            result = result.split(" ")
            result[0] = float(result[0])
            result[1] = float(result[1])
            if(tpb == 32):
                cpu.append(result[1])
            gpu.append(result[0])
            iterations_list.append(iterations)
            particles_list.append(particles)
            iterations = iterations * 10
        particles = particles * 10
        iterations = 2000
    plt.plot(particles_list, gpu, label = "GPU with TPB = {}".format(tpb))
    print(gpu)
    if(tpb == 32):
        plt.plot(particles_list, cpu, label = "CPU")
    particles = 1000
    iterations = 2000
    tpb = tpb*2

plt.xlabel("Number of particles")
plt.ylabel("Seconds")

plt.legend()
plt.savefig("fig2.png")
plt.show()
    
