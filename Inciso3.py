from dynamicDiffusion import DynamicDiffusion
import numpy as np
import os
import matplotlib.pyplot as plt

def experimentar_variacion_K(M, N, T, u0, valores_K):
    for K in valores_K:
        print(f"Ejecutando simulación con K={K}")
        sim = DynamicDiffusion(M, N, T, u0, K)
        sim.runSimulation()
        sim.plotDiffusion()
        name = f'diffusion_K_{int(K*10)}'
        save_png_images(sim, name)

def experimentar_variacion_distribucion_inicial(M, N, T, K, distribuciones):
    for idx, u0 in enumerate(distribuciones):
        print(f"Ejecutando simulación con distribución inicial {idx + 1}")
        sim = DynamicDiffusion(M, N, T, u0, K)
        sim.runSimulation()
        sim.plotDiffusion()
        name = f'diffusion_u0_{idx + 1}'
        save_png_images(sim, name)

def generar_distribuciones_iniciales(M, N):
    distribuciones = []

    # Zonas específicas
    u0_1 = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            if 5 <= i <= 15 and 5 <= j <= 15:
                u0_1[i, j] = 1
            if 40 <= i <= 50 and 10 <= j <= 20:
                u0_1[i, j] = 1
            if 5 <= i <= 15 and 40 <= j <= 50:
                u0_1[i, j] = 1
            if 10 <= i <= 20 and 70 <= j <= 80:
                u0_1[i, j] = 1
            if 70 <= i <= 80 and 50 <= j <= 60:
                u0_1[i, j] = 1
    distribuciones.append(u0_1)

    # Distribución uniforme
    u0_2 = np.ones((M, N))
    distribuciones.append(u0_2)

    # Distribución aleatoria
    u0_3 = np.random.rand(M, N)
    distribuciones.append(u0_3)

    return distribuciones

def save_png_images(sim, name, intervalos=[0, 25, 50, 75, 99]):
    fig, ax = plt.subplots()
    cax = ax.matshow(sim.history[0], cmap='viridis')
    fig.colorbar(cax)

    for frame in intervalos:
        cax.set_data(sim.history[frame])  # Actualiza los datos del visualizador
        ax.set_title(f"Tiempo: {frame}")

        # Guardar la imagen como PNG
        png_path = f'files/Inciso3/{name}_t{frame}.png'
        plt.savefig(png_path)
        print(f"Imagen guardada en {png_path}")
    plt.close(fig)

if __name__ == "__main__":

    # Crear directorio para guardar los PNGs
    output_dir = 'files/Inciso3'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parámetros de la simulación
    M, N = 20, 20
    T = 100
    K_fijo = 0.2  # Valor de K fijo para variar distribución inicial
    valores_K = [0.1, 0.2, 0.3, 0.5]  # Diferentes valores de K

    # Generar distribuciones iniciales
    distribuciones = generar_distribuciones_iniciales(M, N)

    # Variando K con una distribución inicial fija
    print("Variando K con distribución inicial fija")
    experimentar_variacion_K(M, N, T, distribuciones[0], valores_K)

    # Variando la distribución inicial con un K fijo
    print("Variando distribución inicial con K fijo")
    experimentar_variacion_distribucion_inicial(M, N, T, K_fijo, distribuciones)
