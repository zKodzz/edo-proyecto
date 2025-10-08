import numpy as np

def euler_explicito(f, h, x0, y0, xmax):
    """Euler explícito para y' = f(x,y). Devuelve (xs, ys)."""
    n = int(round((xmax - x0) / h))
    xs = x0 + h * np.arange(n + 1)
    ys = np.empty(n + 1, dtype=float)
    ys[0] = y0
    for k in range(n):
        # y_{n+1} = y_n + h f(x_n, y_n)
        ys[k+1] = ys[k] + h * f(xs[k], ys[k])
    return xs, ys

def euler_implicito_lineal(lmbda, h, x0, y0, xmax):
    """Backward Euler para y' = λ y. Devuelve (xs, ys)."""
    n = int(round((xmax - x0) / h))
    xs = x0 + h * np.arange(n + 1)
    ys = np.empty(n + 1, dtype=float)
    ys[0] = y0
    denom = 1.0 - lmbda * h  # para λ=-10, queda 1 + 10h
    for k in range(n):
        # y_{n+1} = y_n / (1 - λ h)
        ys[k+1] = ys[k] / denom
    return xs, ys
