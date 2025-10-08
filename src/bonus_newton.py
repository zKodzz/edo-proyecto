
import numpy as np

def sech2(y):
    c = np.cosh(y)
    return 1.0 / (c*c)

def newton_bisection(G, dG, a, b, y0, tol=1e-10, maxit=60):
    """Newton con bisección segura en [a,b]. Requiere G(a)*G(b) <= 0 y G' > 0 en el rango."""
    ya = G(a); yb = G(b)
    if not np.isfinite(ya) or not np.isfinite(yb) or ya*yb > 0:
        return None, False
    y = float(np.clip(y0, a, b))
    for _ in range(maxit):
        Gy = G(y)
        dGy = dG(y)
        if not np.isfinite(Gy) or not np.isfinite(dGy) or dGy <= 0:
            # cae a bisección si derivada mala
            mid = 0.5*(a+b)
            y = mid
        else:
            # paso de Newton proyectado al intervalo
            y_new = y - Gy/dGy
            if y_new < a or y_new > b or not np.isfinite(y_new):
                y_new = 0.5*(a+b)
            y = y_new

        Gy = G(y)
        if abs(Gy) <= tol*(1.0+abs(y)):
            return y, True

        # actualizar intervalo por signo (G es creciente)
        if Gy > 0:
            b = y
        else:
            a = y
        if abs(b - a) <= tol*(1.0+abs(y)):
            return 0.5*(a+b), True

    return y, False

def backward_euler_nonlinear(f, fy, h, x0, y0, xmax):
    n = int(round((xmax - x0) / h))
    xs = x0 + h * np.arange(n + 1)
    ys = np.empty(n + 1, dtype=float)
    ys[0] = y0
    for k in range(n):
        xk = xs[k]
        yk = ys[k]
        xkp = xs[k+1]

        # G(y) = y - yk - h f(x_{k+1}, y)
        G  = lambda y: y - yk - h*f(xkp, y)
        dG = lambda y: 1.0 - h*fy(xkp, y)

        # Semilla: predictor explícito
        y_pred = yk + h*f(xk, yk)

        # Bracket seguro usando monotonicidad (h <= 0.25 ⇒ dG >= 0.5)
        M = 10.0 + abs(yk) + abs(y_pred)
        a, b = yk - M, yk + M
        root, ok = newton_bisection(G, dG, a, b, y_pred, tol=1e-10, maxit=80)
        if not ok or not np.isfinite(root):
            raise RuntimeError(f"No convergió en paso {k} (h={h}).")
        ys[k+1] = float(root)
    return xs, ys
