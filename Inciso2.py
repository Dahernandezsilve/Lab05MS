import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio
import os

class ParticleDiffusion:
    def __init__(self, M, N, P, T, u0, K, Nexp, mask=None):
        self.M = M
        self.N = N
        self.P = P
        self.T = T
        self.u0 = u0
        self.K = K
        self.Nexp = Nexp
        self.mask = mask if mask is not None else np.ones((M, N), dtype=bool)
        self.particles = []
        self.vecinos = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        self.history = np.zeros((T, M, N))  # Almacenar la media de las partículas en cada celda
        print("🔧 Simulación de partículas inicializada")

    def init_particles(self):
        u_flat = self.u0.flatten()
        positions = np.random.choice(self.M * self.N, size=self.P, p=u_flat)
        self.particles = np.array([divmod(pos, self.N) for pos in positions])

    def run_simulation(self):
        for exp in range(self.Nexp):
            self.init_particles()
            grid_count = np.zeros((self.M, self.N))
            for t in range(self.T):
                new_positions = []
                for i, (x, y) in enumerate(self.particles):
                    if np.random.rand() < self.K:  # Mover partícula con probabilidad K
                        dx, dy = self.vecinos[np.random.randint(0, 8)]
                        new_x, new_y = x + dx, y + dy
                        if 0 <= new_x < self.M and 0 <= new_y < self.N and self.mask[new_x, new_y]:
                            new_positions.append((new_x, new_y))
                        else:
                            new_positions.append((x, y))
                    else:
                        new_positions.append((x, y))
                self.particles = np.array(new_positions)
                for x, y in self.particles:
                    grid_count[x, y] += 1
                self.history[t] += grid_count
            print(f"🔄 Repetición {exp + 1}/{self.Nexp} completada")
        
        # Normalizar el resultado final (promedio espacial de las simulaciones)
        self.history /= (self.Nexp * self.P)
        print("✅ Simulación de partículas finalizada")

    def plot_diffusion(self, interval=100, save_video=False, save_images=False):
        fig, ax = plt.subplots()
        cax = ax.matshow(self.history[0], cmap='viridis')
        fig.colorbar(cax)

        def update(frame):
            cax.set_data(self.history[frame])
            ax.set_title(f"Tiempo: {frame}")
            return cax,

        ani = FuncAnimation(fig, update, frames=self.T, interval=interval, blit=True)

        # Guardar el video si es requerido
        if save_video:
            video_path = 'files/Inciso2/particle_diffusion.mp4'
            ani.save(video_path, writer='ffmpeg', fps=10)
            print(f"🎥 Video guardado en {video_path}")

        # Guardar imágenes en intervalos de tiempo específicos
        if save_images:
            image_times = [0, 25, 50, 99]
            output_dir = 'files/Inciso2'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for t in image_times:
                plt.matshow(self.history[t], cmap='viridis')
                plt.title(f"Tiempo: {t}")
                plt.colorbar()
                img_path = os.path.join(output_dir, f"diffusion_t{t}.png")
                plt.savefig(img_path)
                print(f"🖼️ Imagen guardada en {img_path}")
        
        plt.show()
        print("📊 Animación generada")

# Ejemplo de uso
if __name__ == "__main__":
    # Parámetros de la simulación
    M, N = 20, 20
    P = 500
    T = 100
    K = 0.1
    Nexp = 50
    u0 = np.zeros((M, N))
    
    # Distribución inicial en forma de anillo
    for i in range(M):
        for j in range(N):
            if 5 <= np.sqrt((i - M//2)**2 + (j - N//2)**2) <= 7:
                u0[i, j] = 1
    u0 /= u0.sum()

    # Máscara de celdas activas
    mask = np.ones((M, N), dtype=bool)
    mask[:M//4, :N//4] = False  # Ejemplo de subgrid removido (región en forma de L)
    
    sim = ParticleDiffusion(M, N, P, T, u0, K, Nexp, mask=mask)
    sim.run_simulation()

    # Generar el video y guardar capturas
    sim.plot_diffusion(save_video=True, save_images=True)
