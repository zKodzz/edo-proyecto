
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .bonus_newton import backward_euler_nonlinear, sech2

# Bonus: y' = y + tanh(y),  y(0)=0,  x∈[0,1]
# Implícito BE: G(y) = y - y_n - h(y + tanh(y)) = 0 (sin forma cerrada razonable)

def f(x, y):
    return y + np.tanh(y)

def fy(x, y):
    # d/dy [y + tanh(y)] = 1 + sech^2(y)
    return 1.0 + sech2(y)

def reference_solution(f, x0, y0, xmax, h_ref=1e-5):
    # Euler explícito con paso muy chico como referencia
    n = int(round((xmax - x0) / h_ref))
    xs = x0 + h_ref * np.arange(n + 1)
    ys = np.empty(n + 1, dtype=float)
    ys[0] = y0
    for k in range(n):
        ys[k+1] = ys[k] + h_ref * f(xs[k], ys[k])
    return xs, ys

def interp_ref(xs_ref, ys_ref, xs):
    return np.interp(xs, xs_ref, ys_ref)

def run_bonus():
    x0, y0, xmax = 0.0, 0.0, 1.0
    hs = [0.25, 0.125, 0.0625]   # h ≤ 0.25 asegura G' > 0 (bracketing limpio)
    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    xs_ref, ys_ref = reference_solution(f, x0, y0, xmax, h_ref=1e-5)

    for h in hs:
        xs_b, ys_b = backward_euler_nonlinear(f, fy, h, x0, y0, xmax)
        y_ref_on_grid = interp_ref(xs_ref, ys_ref, xs_b)

        plt.figure()
        plt.plot(xs_ref, ys_ref, label="Ref. (h=1e-5)")
        plt.plot(xs_b, ys_b, 'o--', label=f"BE (h={h})")
        plt.xlabel("x"); plt.ylabel("y"); plt.title("Bonus: EDO no lineal (y + tanh y)")
        plt.grid(True, alpha=0.3); plt.legend()
        plt.savefig(out_dir / f"bonus_nonlineal_h_{str(h).replace('.', 'p')}.png", dpi=160, bbox_inches='tight')
        plt.close()

    print("Bonus generado en:", out_dir)

if __name__ == "__main__":
    run_bonus()
