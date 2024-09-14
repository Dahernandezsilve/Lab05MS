import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class DynamicDiffusion:
    def __init__(self, M, N, T, u0, K, neigh='8', mask=None):
        self.M = M
        self.N = N
        self.T = T
        self.u0 = u0
        self.K = K
        self.neigh = neigh
        self.mask = mask if mask is not None else np.ones((M, N), dtype=bool)
        self.vecinos = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        self.history = []
        print("🔧 Simulación inicializada")

    def runSimulation(self):
        u = self.u0.copy()
        self.history.append(u.copy())
        for t in range(self.T):
            uNew = u.copy()
            for i in range(self.M):
                for j in range(self.N):
                    if self.mask[i, j]:
                        sumaVecinos = 0
                        for di, dj in self.vecinos:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.M and 0 <= nj < self.N and self.mask[ni, nj]:
                                sumaVecinos += u[ni, nj]
                        uNew[i, j] = (1 - self.K) * u[i, j] + (self.K / 8) * sumaVecinos
            u = uNew
            self.history.append(u.copy())
            print(f"🔄 Iteración {t + 1}/{self.T} completada")
        print("✅ Simulación finalizada")

    def plotDiffusion(self, interval=100):
        fig, ax = plt.subplots()
        cax = ax.matshow(self.history[0], cmap='viridis')
        fig.colorbar(cax)

        def update(frame):
            cax.set_data(self.history[frame])
            ax.set_title(f"Tiempo: {frame}")
        
        ani = FuncAnimation(fig, update, frames=len(self.history), interval=interval)
        plt.show()
        print("📊 Animación generada")


# Ejemplo de uso de la clase DynamicDiffusion
if __name__ == "__main__":
    
    # Parámetros de la simulación
    M, N = 20, 20
    T = 100
    K = 0.1
    u0 = np.zeros((M, N))
    # Distribución inicial en forma de anillo
    for i in range(M):
        for j in range(N):
            if 5 <= np.sqrt((i - M//2)**2 + (j - N//2)**2) <= 7:
                u0[i, j] = 1
    u0 /= u0.sum()

    # Máscara de celdas activas
    mask = np.zeros((M, N), dtype=bool)  
    mask[:M//2, :N//2] = True
    mask[M//2:, :N//2] = True
    mask[M//2:, N//2:] = True

    # Uso de la clase DynamicDiffusion
    sim = DynamicDiffusion(M, N, T, u0, K, mask=mask)
    sim.runSimulation()
    sim.plotDiffusion()    
