import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# ÍTEM 1: DIAGRAMA DE TELARAÑA (COBWEB PLOT)
# ==========================================
def cobweb_plot(rho, L, x0, n_max=80):
    """
    Genera y despliega el Diagrama de Telaraña para el modelo logístico discreto.
    
    Permite evaluar visualmente la estabilidad de los puntos fijos del sistema
    y observar fenómenos de convergencia estable o la aparición de oscilaciones
    inestables en función de los parámetros asignados.
    
    Parámetros:
        rho (float): Tasa intrínseca de crecimiento poblacional.
        L (float): Capacidad de carga del entorno.
        x0 (float): Población o condición inicial en el tiempo t=0.
        n_max (int): Número máximo de iteraciones temporales a simular.
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
    Calcula de manera iterativa el tiempo (en número de pasos) que le toma 
    al sistema biológico alcanzar el estado estacionario o capacidad de carga L.
    
    Parámetros:
        rho (float): Tasa intrínseca de crecimiento.
        L (float): Capacidad de carga o límite poblacional.
        x0 (float): Población inicial.
        tol (float): Tolerancia matemática para definir el criterio de parada.
        max_iter (int): Límite superior de iteraciones para evitar bucles infinitos.
        
    Retorna:
        int: Número de iteraciones requerido para converger bajo la tolerancia dada.
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
    Construye y visualiza una matriz bidimensional (mapa de calor) para analizar
    la sensibilidad del tiempo de convergencia frente a cambios simultáneos
    en la tasa de crecimiento (rho) y la capacidad de carga (L).
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
    Evalúa la solución analítica exacta de la ecuación diferencial logística continua.
    
    Parámetros:
        t (ndarray): Vector de tiempo.
        r (float): Coeficiente de crecimiento continuo.
        L (float): Capacidad de carga del ecosistema.
        X0 (float): Población inicial continua.
    """
    return (L * X0) / (X0 + (L - X0) * np.exp(-r * t))

def euler_integracion(r, L, X0, h, t_max=25):
    """
    Aproxima la solución de la ecuación diferencial mediante el método numérico de Euler.
    
    Permite estudiar de forma cuantitativa cómo se propaga el error global 
    en función del tamaño de paso h seleccionado.
    
    Parámetros:
        r (float): Coeficiente de crecimiento.
        L (float): Capacidad de carga.
        X0 (float): Población inicial.
        h (float): Tamaño de paso temporal (step-size).
        t_max (float): Tiempo máximo de la simulación.
    """
    n_pasos = int(t_max / h)
    t_vals = np.linspace(0, t_max, n_pasos + 1)
    X_vals = np.zeros(n_pasos + 1)
    X_vals[0] = X0
    for n in range(n_pasos):
        X_vals[n+1] = X_vals[n] + h * (r * X_vals[n] * (1 - X_vals[n] / L))
    return t_vals, X_vals

def graficar_error_euler():
    """
    Genera la gráfica comparativa institucional entre la curva exacta (solución analítica)
    y los múltiples perfiles numéricos derivados del método de Euler para h arbitrarios.
    """
    r, L, X0 = 0.1, 1000, 100
    t_max = 25
    pasos_h = [1.0, 0.5, 0.25, 0.1]
    
    plt.figure(figsize=(10, 5))
    t_analitico = np.linspace(0, t_max, 500)
    X_analitico = solucion_analitica(t_analitico, r, L, X0)
    plt.plot(t_analitico, X_analitico, 'k-', lw=2, label='Solución Analítica (Exacta)')
    
    for h in pasos_h:
        t_e, X_e = euler_integracion(r, L, X0, h, t_max)
        plt.plot(t_e, X_e, '--', label=f'Euler numérico ($h={h}$)')
        
    plt.title('Comparación: Solución Analítica vs. Método de Euler')
    plt.xlabel('Tiempo ($t$)')
    plt.ylabel('Población ($X$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
