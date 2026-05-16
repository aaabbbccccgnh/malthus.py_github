import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# ÍTEM 1: DIAGRAMA DE TELARAÑA (COBWEB PLOT)
# ==========================================
def cobweb_plot(rho, L, x0, n_max=80):
    # Modelo discreto del enunciado: X_{n+1} = X_n + rho * X_n * (L - X_n)
    f = lambda x: x + rho * x * (L - x)
    x_vals = np.linspace(0, L * 1.1, 1000)
    y_vals = f(x_vals)
    
    plt.figure(figsize=(6, 6))
    plt.plot(x_vals, y_vals, 'b', lw=1.5, label=r'$f(X_n) = X_n + \rho X_n(L - X_n)$')
    plt.plot(x_vals, x_vals, 'g--', alpha=0.7, label='$Y = X$')
    
    x_curr = x0
    for _ in range(n_max):
        y_next = f(x_curr)
        plt.plot([x_curr, x_curr], [x_curr, y_next], 'r', alpha=0.5, lw=1)
        plt.plot([x_curr, y_next], [y_next, y_next], 'r', alpha=0.5, lw=1)
        x_curr = y_next
        if abs(x_curr) > 1e4: break
            
    plt.title(f'Diagrama de Telaraña - r={rho}, L={L}')
    plt.xlabel('$X_n$')
    plt.ylabel('$X_{n+1}$')
    plt.xlim(0, L * 1.1)
    plt.ylim(0, L * 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ===================================================
# ÍTEM 2: ANÁLISIS DE SENSIBILIDAD Y TIEMPO CONVERGENCIA
# ===================================================
def calcular_tiempo_convergencia(rho, L, x0=5, tol=1e-2, max_iter=1000):
    f = lambda x: x + rho * x * (L - x)
    x = x0
    for t in range(max_iter):
        x_next = f(x)
        # Verifica si la población dejó de cambiar drásticamente y llegó cerca de L
        if abs(x_next - x) < tol and abs(x_next - L) < tol * 10:
            return t
        x = x_next
        if x <= 0 or np.isnan(x): return max_iter
    return max_iter


# =========================================
# ÍTEM 3: ESTUDIO DEL ERROR GLOBAL DE EULER
# =========================================
def solucion_analitica(t, r, L, X0):
    return (L * X0) / (X0 + (L - X0) * np.exp(-r * t))

def euler_integracion(r, L, X0, h, t_max=25):
    n_pasos = int(t_max / h)
    t_vals = np.linspace(0, t_max, n_pasos + 1)
    X_vals = np.zeros(n_pasos + 1)
    X_vals[0] = X0
    for n in range(n_pasos):
        # Algoritmo numérico de Euler: X_{n+1} = X_n + h * f(t_n, X_n)
        X_vals[n+1] = X_vals[n] + h * (r * X_vals[n] * (1 - X_vals[n] / L))
    return t_vals, X_vals


# =====================================================================
# EJECUCIÓN AUTOMÁTICA DE LOS GRÁFICOS (BLOQUE DE PRUEBA LOCAL)
# =====================================================================
if __name__ == "__main__":
    print("Generando Gráfico A: Convergencia con rho pequeño...")
    cobweb_plot(rho=0.01, L=100, x0=10) 

    print("Generando Gráfico B: Oscilaciones con rho grande...")
    cobweb_plot(rho=0.026, L=100, x0=10)
