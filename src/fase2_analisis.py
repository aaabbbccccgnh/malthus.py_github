import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# ÍTEM 1: DIAGRAMA DE TELARAÑA (COBWEB PLOT)
# ==========================================
def cobweb_plot(rho, L, x0, n_max=80):
    """
    Genera y despliega el Diagrama de Telaraña para el modelo logístico discreto.
    """
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
        if abs(x_curr) > 1e4: 
            break
            
    plt.title(f'Diagrama de Telaraña - r={rho}, L={L}')
    plt.xlabel('$X_n$')
    plt.ylabel('$X_{n+1}$')
    plt.xlim(0, L * 1.1)
    plt.ylim(0, L * 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ===================================================
# ÍTEM 2: ANÁLISIS DE SENSIBILIDAD PARAMÉTRICA
# ===================================================
def calcular_tiempo_convergencia(rho, L, x0=5, tol=1e-2, max_iter=1000):
    """
    Calcula el tiempo iterativo para alcanzar el estado estacionario L.
    """
    f = lambda x: x + rho * x * (L - x)
    x = x0
    for t in range(max_iter):
        x_next = f(x)
        if abs(x_next - x) < tol and abs(x_next - L) < tol * 10:
            return t
        x = x_next
        if x <= 0 or np.isnan(x): 
            return max_iter
    return max_iter

def generar_mapa_calor_sensibilidad():
    """
    Visualiza la matriz bidimensional para analizar la sensibilidad de rho y L.
    """
    rho_vals = np.linspace(0.005, 0.025, 50)
    L_vals = np.linspace(50, 150, 50)
    T_matrix = np.zeros((len(rho_vals), len(L_vals)))
    
    for i, rho in enumerate(rho_vals):
        for j, L in enumerate(L_vals):
            T_matrix[i, j] = calcular_tiempo_convergencia(rho, L, x0=10)
            
    plt.figure(figsize=(8, 6))
    plt.imshow(T_matrix, extent=[L_vals[0], L_vals[-1], rho_vals[0], rho_vals[-1]], 
               origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Tiempo de convergencia (iteraciones $t$)')
    plt.title('Análisis de Sensibilidad: Tiempo de Convergencia según $\\rho$ y $L$')
    plt.xlabel('Capacidad de carga ($L$)')
    plt.ylabel('Tasa de crecimiento ($\\rho$)')
    plt.grid(False)
    plt.show()


# =========================================
# ÍTEM 3: ESTUDIO DEL ERROR GLOBAL DE EULER
# =========================================
def solucion_analitica(t, r, L, X0):
    """
    Evalúa la solución analítica exacta de la ecuación diferencial logística.
    """
    return (L * X0) / (X0 + (L - X0) * np.exp(-r * t))

def euler_integracion(r, L, X0, h, t_max=25):
    """
    Aproxima la solución de la ecuación diferencial mediante el método de Euler.
    """
    n_pasos = int(t_max / h)
    t_vals = np.linspace(0, t_max, n_pasos + 1)
    X_vals = np.zeros(n_pasos + 1)
    X_vals[0] = X0
    for n in range(n_pasos):
        X_vals[n+1] = X_vals[n] + h * (r * X_vals[n] * (1 - X_vals[n] / L))
    return t_vals, X_vals

# Esto activará los gráficos de telaraña que programaron
cobweb_plot(rho=0.01, L=100, x0=10)   # Caso A: Estable
cobweb_plot(rho=0.026, L=100, x0=10)  # Caso B: Inestable

# ===================================================
# EJECUTAR ÍTEM 2: TIEMPOS DE CONVERGENCIA (TABLA)
# ===================================================
print("--- TIEMPOS DE CONVERGENCIA ---")
for r in [0.005, 0.01, 0.015]:
    t = calcular_tiempo_convergencia(rho=r, L=100, x0=5)
    print(f"Para rho = {r}, el tiempo de convergencia es: {t} pasos.")

print("\n" + "="*40 + "\n")

# ===================================================
# EJECUTAR ÍTEM 3: GRÁFICO DE ERROR DE EULER
# ===================================================
print("--- GENERANDO GRÁFICO DE EULER ---")

# Parámetros de prueba
r_prueba = 0.1
L_prueba = 100
X0_prueba = 10
t_max_prueba = 25

plt.figure(figsize=(8, 5))

# 1. Solución analítica (continua) como referencia
t_exacto = np.linspace(0, t_max_prueba, 200)
X_exacto = solucion_analitica(t_exacto, r_prueba, L_prueba, X0_prueba)
plt.plot(t_exacto, X_exacto, 'k-', label='Solución Exacta', linewidth=2)

# 2. Probar Euler con diferentes pasos (h) para ver el error
pasos_h = [2.0, 0.5, 0.1]
for h in pasos_h:
    t_euler, X_euler = euler_integracion(r_prueba, L_prueba, X0_prueba, h, t_max_prueba)
    plt.plot(t_euler, X_euler, '--o', label=f'Euler (h = {h})', alpha=0.7)

plt.title('Comparación: Método de Euler vs Solución Exacta')
plt.xlabel('Tiempo (t)')
plt.ylabel('Población (X)')
plt.legend()
plt.grid(True)
plt.show()
