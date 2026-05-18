import numpy as np
import matplotlib.pyplot as plt
import os

def generar_diagrama_bifurcacion(n_mu=2000, iteraciones=1000, ultimos=100):
    """
    Simula el mapa logístico y guarda el diagrama de bifurcación
    directamente en la carpeta 'figs' sin abrir ventanas emergentes.
    """
    mu_valores = np.linspace(0.0, 4.0, n_mu)
    x_grafico = []
    y_grafico = []

    for mu in mu_valores:
        x = 0.5  # Condición inicial
        
        for i in range(iteraciones):
            x = mu * x * (1 - x)
            if i >= (iteraciones - ultimos):
                x_grafico.append(mu)
                y_grafico.append(x)

    # 1. Configuración del gráfico (con 'r' para LaTeX)
    plt.figure(figsize=(12, 8))
    plt.plot(x_grafico, y_grafico, ',k', alpha=0.2)
    
    plt.title(r'Diagrama de Bifurcación: Mapa Logístico y Caos ($\mu \in [0,4]$)')
    plt.xlabel(r'Parámetro de crecimiento ($\mu$)')
    plt.ylabel(r'Estado estacionario de la población ($x$)')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 4)
    plt.ylim(0, 1)
    
    # 2. SISTEMA DE GUARDADO EXPLÍCITO (Requisito del Proyecto)
    # Crea la carpeta "figs" automáticamente si no existe en tu directorio
    if not os.path.exists('figs'):
        os.makedirs('figs')
        
    ruta_guardado = 'figs/diagrama_bifurcacion.png'
    
    # Guarda el archivo en alta definición (dpi=300) y ajusta los márgenes
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    
    # IMPORTANTE: Cerramos el gráfico en la memoria de Python. 
    # Esto evita por completo que se intente levantar una ventana flotante.
    plt.close() 
    
    print(f"--> ¡Cálculo completado con éxito!")
    print(f"--> Gráfico exportado de forma explícita en: {ruta_guardado}")

if __name__ == "__main__":
    print("Ejecutando simulación en src/modelos.py...")
    generar_diagrama_bifurcacion()