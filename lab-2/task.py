from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from sympy import Symbol, sympify
from sympy.utilities.lambdify import lambdify


@dataclass
class OptimizationResult:
    x_min: float
    f_min: float
    iterations: int
    elapsed: float
    samples: List[Tuple[float, float]]


def build_function(
    expression: str,
) -> Tuple[Callable[[float], float], Callable[[np.ndarray], np.ndarray]]:
    x = Symbol("x")
    expr = sympify(expression, locals={"pi": math.pi})
    f_numpy = lambdify(x, expr, modules=["numpy"])

    def f_scalar(value: float) -> float:
        return float(f_numpy(value))

    return f_scalar, f_numpy


def strongin_global_search(
    func: Callable[[float], float],
    interval: Tuple[float, float],
    eps: float,
    r: float = 2.0,
    max_iter: int = 10_000,
) -> OptimizationResult:
    """Реализация однопараметрического метода Строгина."""

    left, right = interval
    if left >= right:
        raise ValueError("Левая граница должна быть меньше правой.")

    if eps <= 0:
        raise ValueError("Точность eps должна быть положительной.")

    start_time = time.perf_counter()
    sampled = [(left, func(left)), (right, func(right))]
    iterations = 0

    while iterations < max_iter:
        sampled.sort(key=lambda p: p[0])

        slopes = []
        for (x0, y0), (x1, y1) in zip(sampled[:-1], sampled[1:]):
            delta = x1 - x0
            if delta <= 0:
                continue
            slopes.append(abs(y1 - y0) / delta)

        m = max(slopes) if slopes else 0.0
        m = r * m if m > 0 else 1.0

        best_idx = None
        best_r = -math.inf

        for i in range(len(sampled) - 1):
            xi, yi = sampled[i]
            xj, yj = sampled[i + 1]
            delta = xj - xi
            if delta <= 0:
                continue

            r_i = m * delta + (yj - yi) ** 2 / (m * delta) - 2.0 * (yj + yi)

            if r_i > best_r:
                best_r = r_i
                best_idx = i

        if best_idx is None:
            break

        xi, yi = sampled[best_idx]
        xj, yj = sampled[best_idx + 1]
        delta = xj - xi

        if delta < eps:
            break

        x_new = 0.5 * (xi + xj) - (yj - yi) / (2.0 * m)
        x_new = min(max(x_new, left), right)
        y_new = func(x_new)

        sampled.append((x_new, y_new))
        iterations += 1

    sampled.sort(key=lambda p: p[0])
    x_min, f_min = min(sampled, key=lambda p: p[1])
    elapsed = time.perf_counter() - start_time

    return OptimizationResult(
        x_min=x_min,
        f_min=f_min,
        iterations=iterations,
        elapsed=elapsed,
        samples=sampled,
    )


def render_report(
    func_vectorized: Callable[[np.ndarray], np.ndarray],
    result: OptimizationResult,
    interval: Tuple[float, float],
    expression: str,
    output_path: Path,
    title: str = "Глобальный поиск минимума",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_min, x_max = interval
    xs = np.linspace(x_min, x_max, 1_000)
    ys = func_vectorized(xs)

    samples = np.array(result.samples)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    ax.plot(xs, ys, label="f(x)", color="#1f77b4", linewidth=2)
    ax.plot(
        samples[:, 0],
        samples[:, 1],
        marker="o",
        linestyle="-",
        color="#ff7f0e",
        label="Ломаная по испытательным точкам",
    )
    ax.scatter(
        [result.x_min],
        [result.f_min],
        color="#2ca02c",
        s=80,
        zorder=5,
        label="Найденный минимум",
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")

    info_lines = [
        f"f(x) = {expression}",
        f"[a, b] = [{x_min:.6g}, {x_max:.6g}]",
        f"x* ≈ {result.x_min:.6g}",
        f"f(x*) ≈ {result.f_min:.6g}",
        f"Количество итераций: {result.iterations}",
        f"Время работы: {result.elapsed:.3f} с",
        f"Число испытательных точек: {len(result.samples)}",
    ]
    info_text = "\n".join(info_lines)
    fig.text(
        0.98,
        0.02,
        info_text,
        fontsize=10,
        ha="right",
        va="bottom",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")

    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Поиск глобального экстремума одномерной липшицевой функции.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-2,
        help="Точность остановки (по x).",
    )
    parser.add_argument(
        "--r",
        type=float,
        default=2.0,
        help="Параметр надежности метода Строгина (r > 1).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=10_000,
        help="Максимальное число итераций.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lab-2") / "optimization_report.pdf",
        help="Путь к итоговому PDF-отчету (без расширения PNG).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Запустить демонстрацию на тестовой функции Растригина.",
    )
    return parser.parse_args(argv)


def run_demo(args: argparse.Namespace) -> None:
    expression = "x**2 - 10*cos(2*pi*x) + 10"
    left, right = -5.12, 5.12

    f_scalar, f_vector = build_function(expression)
    result = strongin_global_search(
        f_scalar,
        (left, right),
        eps=args.eps,
        r=args.r,
        max_iter=args.max_iter,
    )
    render_report(
        f_vector,
        result,
        (left, right),
        expression,
        Path(args.output),
        title="Демонстрация: функция Растригина (1D)",
    )

def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    run_demo(args)


if __name__ == "__main__":
    main()

