from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

_allowed_math_names = {k: getattr(math, k) for k in dir(math) if not k.startswith('\n')}
_allowed_math_names.update({'pi': math.pi, 'e': math.e, 'inf': math.inf})


def make_function_from_string(expr: str) -> Callable[[float], float]:
    local_dict = dict(_allowed_math_names)  # copy
    code = compile(expr, '<string>', 'eval')
    def f(x):
        local_dict['x'] = float(x)
        return float(eval(code, {'__builtins__': {}}, local_dict))
    return f


def make_vectorized_function(f: Callable[[float], float]) -> Callable[[np.ndarray], np.ndarray]:
    """Создаёт векторизованную версию функции для работы с numpy массивами."""
    def f_vectorized(x_arr: np.ndarray) -> np.ndarray:
        return np.array([f(x) for x in x_arr])
    return f_vectorized


def piyavskii(
    f: Callable[[float], float],
    a: float,
    b: float,
    eps: float = 1e-2,
    L: float = None,
    max_iter: int = 20000,
    initial_points: int = 5,
    verbose: bool = False
) -> dict:
    t0 = time.perf_counter()
    xs = list(np.linspace(a, b, initial_points))
    fs = [f(x) for x in xs]
    
    if L is None:
        L_est = 0.0
        for i in range(len(xs)-1):
            dx = xs[i+1] - xs[i]
            slope = abs(fs[i+1] - fs[i]) / dx
            if slope > L_est:
                L_est = slope
        L = max(L_est * 1.2, 1e-6)
        if verbose:
            print(f'Estimated L = {L:.6g}')
    else:
        L = float(L)

    def intersection_x(xi, fi, xj, fj):
        return (fj - fi) / (2*L) + (xj + xi) / 2

    iterations = 0
    history = []
    while iterations < max_iter:
        iterations += 1
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]
        fs = [fs[i] for i in order]
        upper = min(fs)
        x_upper = xs[fs.index(upper)]
        
        best_interval = None
        best_h = math.inf
        best_x = None
        for i in range(len(xs)-1):
            xi, xj = xs[i], xs[i+1]
            fi, fj = fs[i], fs[i+1]
            x_star = intersection_x(xi, fi, xj, fj)
            if x_star <= xi:
                x_star = (xi + xj) / 2.0
            if x_star >= xj:
                x_star = (xi + xj) / 2.0
            h_val = fi - L * (x_star - xi)
            if h_val < best_h:
                best_h = h_val
                best_interval = (xi, xj, fi, fj)
                best_x = x_star
        
        lower = best_h
        
        if upper - lower <= eps:
            t_elapsed = time.perf_counter() - t0
            return {
                'x_min': x_upper,
                'f_min': upper,
                'iterations': iterations,
                'time': t_elapsed,
                'xs': xs,
                'fs': fs,
                'L': L,
                'lower_bound': lower,
                'upper_bound': upper,
                'history': history
            }
        
        fx_new = f(best_x)
        xs.append(best_x)
        fs.append(fx_new)
        history.append((best_x, fx_new))
        
        if iterations % 200 == 0 and iterations > 0:
            L_est = 0.0
            for i in range(len(xs)-1):
                dx = abs(xs[i+1] - xs[i])
                slope = abs(fs[i+1] - fs[i]) / dx if dx > 0 else 0.0
                if slope > L_est:
                    L_est = slope
            if L_est > L:
                if verbose:
                    print(f'Increasing L: {L} -> {L_est*1.05}')
                L = L_est * 1.05

    t_elapsed = time.perf_counter() - t0
    best_f = min(fs)
    best_x = xs[fs.index(best_f)]
    return {
        'x_min': best_x,
        'f_min': best_f,
        'iterations': iterations,
        'time': t_elapsed,
        'xs': xs,
        'fs': fs,
        'L': L,
        'lower_bound': None,
        'upper_bound': best_f,
        'history': history
    }


def compute_lower_envelope(xs, fs, L, grid_x):
    """Вычисляет нижнюю огибающую для визуализации."""
    hvals = np.full_like(grid_x, np.inf, dtype=float)
    for xi, fi in zip(xs, fs):
        hvals = np.minimum(hvals, fi - L * np.abs(grid_x - xi))
    return hvals

def render_report(
    func: Callable[[float], float],
    func_vectorized: Callable[[np.ndarray], np.ndarray],
    result: dict,
    interval: Tuple[float, float],
    expression: str,
    output_path: Path,
    title: str = "Глобальный поиск минимума",
) -> None:
    """Создаёт визуализацию и сохраняет отчёт в PDF с информацией о замерах."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_min, x_max = interval
    xs = np.linspace(x_min, x_max, 1_000)
    ys = func_vectorized(xs)
    
    # Вычисляем нижнюю огибающую
    hvals = compute_lower_envelope(result['xs'], result['fs'], result['L'], xs)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

    # График исходной функции
    ax.plot(xs, ys, label="f(x)", color="#1f77b4", linewidth=2)
    
    # Нижняя огибающая
    ax.plot(xs, hvals, '--', linewidth=1.5, color="#9467bd", label="Нижняя огибающая h(x)")
    
    # Испытанные точки
    xs_arr = np.array(result['xs'])
    fs_arr = np.array(result['fs'])
    ax.scatter(xs_arr, fs_arr, c='red', s=30, zorder=4, label='Испытанные точки', alpha=0.7)
    
    # Найденный минимум
    ax.scatter(
        [result['x_min']],
        [result['f_min']],
        color="#2ca02c",
        s=120,
        zorder=5,
        marker='*',
        edgecolors='black',
        linewidths=1,
        label="Найденный минимум",
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("f(x)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=10)

    info_lines = [
        f"f(x) = {expression}",
        f"[a, b] = [{x_min:.6g}, {x_max:.6g}]",
        f"x* ≈ {result['x_min']:.6g}",
        f"f(x*) ≈ {result['f_min']:.6g}",
        f"Количество итераций: {result['iterations']}",
        f"Время работы: {result['time']:.4f} с",
        f"Число испытательных точек: {len(result['xs'])}",
        f"Оценка константы Липшица L: {result['L']:.6g}",
    ]
    if result.get('lower_bound') is not None:
        info_lines.append(f"Нижняя граница: {result['lower_bound']:.6g}")
        info_lines.append(f"Верхняя граница: {result['upper_bound']:.6g}")
    
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

    # Также сохраняем PNG
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Поиск глобального экстремума одномерной липшицевой функции методом Piyavskii-Shubert.",
    )
    parser.add_argument(
        "--function",
        type=str,
        help="Строка с выражением функции f(x). Если не указана, используется демонстрационная функция.",
    )
    parser.add_argument(
        "--left",
        type=float,
        help="Левая граница отрезка.",
    )
    parser.add_argument(
        "--right",
        type=float,
        help="Правая граница отрезка.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-2,
        help="Точность остановки (по разнице верхней и нижней границ).",
    )
    parser.add_argument(
        "--L",
        type=float,
        default=None,
        help="Константа Липшица (если не указана, оценивается автоматически).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=20000,
        help="Максимальное число итераций.",
    )
    parser.add_argument(
        "--initial-points",
        type=int,
        default=5,
        help="Число начальных точек для инициализации.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lab-2") / "optimization_report.pdf",
        help="Путь к итоговому PDF-отчету (PNG будет создан автоматически).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Запустить демонстрацию на функции Растригина.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Выводить дополнительную информацию во время работы.",
    )
    return parser.parse_args(argv)


def run_custom(args: argparse.Namespace) -> None:
    """Запуск с пользовательскими параметрами."""
    if args.function is None or args.left is None or args.right is None:
        raise ValueError(
            "Для пользовательского запуска требуется задать --function, --left и --right.",
        )

    f_scalar = make_function_from_string(args.function)
    f_vector = make_vectorized_function(f_scalar)
    
    result = piyavskii(
        f_scalar,
        args.left,
        args.right,
        eps=args.eps,
        L=args.L,
        max_iter=args.max_iter,
        initial_points=args.initial_points,
        verbose=args.verbose,
    )
    
    render_report(
        f_scalar,
        f_vector,
        result,
        (args.left, args.right),
        args.function,
        Path(args.output),
        title="Глобальный поиск минимума Метод Пиявского",
    )
    
    # Вывод результатов в консоль
    print("\nРезультаты оптимизации:")
    print(f"  Найденный x_min = {result['x_min']:.6f}")
    print(f"  f(x_min) = {result['f_min']:.6f}")
    print(f"  Итераций = {result['iterations']}")
    print(f"  Время = {result['time']:.4f} с")
    print(f"  Оценка L = {result['L']:.6g}")
    print(f"  Отчёт сохранён: {args.output}")


def run_demo(args: argparse.Namespace) -> None:
    """Демонстрационный запуск на функции Растригина."""
    expression = "10 + x**2 - 10*cos(2*pi*x)"
    left, right = -5.12, 5.12

    f_scalar = make_function_from_string(expression)
    f_vector = make_vectorized_function(f_scalar)
    
    result = piyavskii(
        f_scalar,
        left,
        right,
        eps=args.eps,
        L=args.L,
        max_iter=args.max_iter,
        initial_points=args.initial_points,
        verbose=args.verbose,
    )
    
    render_report(
        f_scalar,
        f_vector,
        result,
        (left, right),
        expression,
        Path(args.output),
        title="Демонстрация: Метод Пиявского",
    )
    
    print("\nРезультаты оптимизации (демонстрация):")
    print(f"  Функция: {expression}")
    print(f"  Интервал: [{left}, {right}]")
    print(f"  Найденный x_min = {result['x_min']:.6f}")
    print(f"  f(x_min) = {result['f_min']:.6f}")
    print(f"  Итераций = {result['iterations']}")
    print(f"  Время = {result['time']:.4f} с")
    print(f"  Оценка L = {result['L']:.6g}")
    if result.get('lower_bound') is not None:
        print(f"  Нижняя граница: {result['lower_bound']:.6g}")
        print(f"  Верхняя граница: {result['upper_bound']:.6g}")
    print(f"  Отчёт сохранён: {args.output}")


def main(argv: Sequence[str] | None = None) -> None:
    """Главная функция."""
    args = parse_arguments(argv)
    if args.demo:
        run_demo(args)
    else:
        run_custom(args)


if __name__ == "__main__":
    main()

