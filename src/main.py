import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .euler import euler_explicito, euler_implicito_lineal

# Caso de la pauta: y' = -10 y,  y(0)=1,  x in [0,3]
def f(x, y):
    return -10.0 * y

def exact(x):
    return np.exp(-10.0 * x)

def max_abs_error(xs, ys):
    return float(np.max(np.abs(exact(xs) - ys)))

def run():
    x0, y0, xmax = 0.0, 1.0, 3.0
    hs = [0.5, 0.2, 0.1, 0.05]     # los que piden
    lmbda = -10.0
    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Comparación exacta vs aproximaciones
    fine = np.linspace(x0, xmax, 1200)
    for h in hs:
        xs_e, ys_e = euler_explicito(f, h, x0, y0, xmax)
        xs_i, ys_i = euler_implicito_lineal(lmbda, h, x0, y0, xmax)

        plt.figure()
        plt.plot(fine, exact(fine), label="Exacta e^{-10x}")
        plt.plot(xs_e, ys_e, 'o--', label=f"Euler explícito (h={h})")
        plt.plot(xs_i, ys_i, 's-.', label=f"Euler implícito (h={h})")
        plt.xlabel("x"); plt.ylabel("y"); plt.title(f"Exacta vs Euler (h={h})")
        plt.grid(True, alpha=0.3); plt.legend()
        plt.savefig(out_dir / f"comparacion_h_{str(h).replace('.', 'p')}.png", dpi=160, bbox_inches='tight')
        plt.close()

    # Curva de error: solo con h estables para el explícito (h<0.2)
    import csv
    hs_for_err = [0.05, 0.1]
    rows = [("h", "error_explicito", "error_implicito")]
    for h in hs_for_err:
        xs_e, ys_e = euler_explicito(f, h, x0, y0, xmax)
        xs_i, ys_i = euler_implicito_lineal(lmbda, h, x0, y0, xmax)
        rows.append((h, max_abs_error(xs_e, ys_e), max_abs_error(xs_i, ys_i)))

    csv_path = out_dir / "errores_vs_h.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        csv.writer(fp).writerows(rows)

    # Error vs h
    hs_vals = [r[0] for r in rows[1:]]
    err_e  = [r[1] for r in rows[1:]]
    err_i  = [r[2] for r in rows[1:]]

    plt.figure()
    plt.plot(hs_vals, err_e, 'o-', label="Error máx (Explícito)")
    plt.plot(hs_vals, err_i, 's-', label="Error máx (Implícito)")
    plt.xlabel("h"); plt.ylabel("Error máximo"); plt.title("Error vs h")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(out_dir / "error_vs_h.png", dpi=160, bbox_inches='tight')
    plt.close()

    # Escala log–log para ver pendiente ~1
    plt.figure()
    plt.loglog(hs_vals, err_e, 'o-', label="Explícito")
    plt.loglog(hs_vals, err_i, 's-', label="Implícito")
    plt.xlabel("h (log)"); plt.ylabel("Error máx (log)"); plt.title("Error vs h (log–log)")
    plt.grid(True, which="both", alpha=0.3); plt.legend()
    plt.savefig(out_dir / "error_vs_h_loglog.png", dpi=160, bbox_inches='tight')
    plt.close()

    print("Listo. Archivos en:", out_dir)

if __name__ == "__main__":
    run()
