# Euler explícito e implícito (y' = -10 y)

**Objetivo**: comparar Euler explícito vs implícito para
`y'=-10y, y(0)=1, x∈[0,3]` y graficar contra la solución exacta.
Además, mirar estabilidad y cómo baja el error con h.

## Requisitos
- Python 3.10+
- `numpy`, `matplotlib`
  (si hace falta: `pip install -r requirements.txt`)

## Ejecutar
Desde la carpeta del proyecto:
```bash
python -m src.main        # corre el caso de la pauta
python -m src.main_bonus  # opcional: BE + Newton en EDO no lineal
```
Las figuras y el CSV quedan en `outputs/`.

## Qué hace cada archivo
- `src/euler.py`:
  - `euler_explicito(f, h, x0, y0, xmax)`
  - `euler_implicito_lineal(lmbda, h, x0, y0, xmax)` (Backward Euler para y'=λy)
- `src/main.py`: arma el experimento (h = 0.5, 0.2, 0.1, 0.05),
  grafica exacta vs aproximaciones y calcula errores (solo con h < 0.2).
- `src/bonus_newton.py` + `src/main_bonus.py`: Backward Euler + Newton para
  `y' = e^y + y^3` (cada paso implícito se resuelve por raíces).

## Notas rápidas para el informe
- Estabilidad de Euler explícito en `y'=-a y`: `|1 - a h| < 1` → `h < 2/a`.
  Con `a=10`, `h < 0.2`. Por eso con `h≥0.2` se empieza a ir a las nubes.
- Ambos métodos son de orden 1 → el error baja ~lineal con h.
