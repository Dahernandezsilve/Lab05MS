from dynamicDiffusion import DynamicDiffusion
import numpy as np

def Inciso1():
    M, N = 100, 100
    T = 100
    K = 0.2
    u0 = np.zeros((M, N))

    for i in range(M):
        for j in range(N):
            if 5 <= i <= 15 and 5 <= j <= 15:
                u0[i, j] = 1
    
    for i in range(M):
        for j in range(N):
            if 40 <= i <= 50 and 10 <= j <= 20:
                u0[i, j] = 1
    
    for i in range(M):
        for j in range(N):
            if 5 <= i <= 15 and 40 <= j <= 50:
                u0[i, j] = 1

    for i in range(M):
        for j in range(N):
            if 10 <= i <= 20 and 70 <= j <= 80:
                u0[i, j] = 1

    for i in range(M):
            for j in range(N):
                if 70 <= i <= 80 and 50 <= j <= 60:
                    u0[i, j] = 1

    sim = DynamicDiffusion(M, N, T, u0, K)
    sim.runSimulation()
    sim.plotDiffusion()

Inciso1()