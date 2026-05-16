import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==========================================
# ÍTEM 1: DIAGRAMA DE TELARAÑA (COBWEB PLOT)
# ==========================================
def cobweb_plot(rho, L, x0, n_max=80):
    f = lambda x: x + rho * x * (L - x)
    x_vals = np.linspace(0, L * 1.1, 1000)
    y_vals = f(x_vals)
    
    plt.figure(figsize=(7, 7))
    plt.plot(x_vals, y_vals, 'b', lw=1.5, label=r'$f(X_n) = X_n + \rho X_n(L - X_n)$')
    plt.plot(x_vals, x_vals, 'g--', alpha=0.7, label='$Y = X$')
    
    x_curr = x0
    for _ in range(n_max):
        y_next = f(x_curr)
        plt.plot([x_curr, x_curr], [x_curr, y_next], 'r', alpha=0.5, lw=1)
        plt.plot([x_curr, y_next], [y_next, y_next], 'r', alpha=0.5, lw=1)
        x_curr = y_next
        if abs(x_curr) > 1e4: break
            
    plt.title(f'Diagrama de Telaraña (Cobweb Plot) - r={rho}, L={L}')
    plt.xlabel('$X_n$')
    plt.ylabel('$X_{n+1}$')
    plt.xlim(0, L * 1.1)
    plt.ylim(0, L * 1.1)
    plt.legend()
    plt.show()

# Para ejecutar localmente: cobweb_plot(rho=0.01, L=100, x0=10)
