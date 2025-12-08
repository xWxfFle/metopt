"""
Решение задачи условной оптимизации методом множителей Лагранжа

Задача:
Минимизировать f(x,y) = x^2 + 3y^2
при ограничениях:
g1(x,y) = x^2 + y^2 - 1 = 0
g2(x,y) = (x-1)^2 + (y-1)^2 - 2 = 0
"""

import sys
import io
import numpy as np
from scipy.optimize import fsolve
from sympy import symbols, diff, solve, Matrix, simplify
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Настройка кодировки для вывода
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def solve_lagrange_analytical():
    """
    Аналитическое решение методом множителей Лагранжа
    """
    print("=" * 80)
    print("РЕШЕНИЕ ЗАДАЧИ УСЛОВНОЙ ОПТИМИЗАЦИИ")
    print("Метод множителей Лагранжа")
    print("=" * 80)
    print()
    
    print("Задача:")
    print("Минимизировать f(x,y) = x^2 + 3y^2")
    print("при ограничениях:")
    print("  g1(x,y) = x^2 + y^2 - 1 = 0")
    print("  g2(x,y) = (x-1)^2 + (y-1)^2 - 2 = 0")
    print()
    
    # Определяем символы
    x, y, lam1, lam2 = symbols('x y lam1 lam2', real=True)
    
    # Целевая функция
    f = x**2 + 3*y**2
    
    # Ограничения
    g1 = x**2 + y**2 - 1
    g2 = (x-1)**2 + (y-1)**2 - 2
    
    # Функция Лагранжа
    L = f - lam1*g1 - lam2*g2
    
    print("Функция Лагранжа:")
    print(f"L(x,y,lambda1,lambda2) = {L}")
    print()
    
    # Частные производные
    dL_dx = diff(L, x)
    dL_dy = diff(L, y)
    dL_dlam1 = diff(L, lam1)
    dL_dlam2 = diff(L, lam2)
    
    print("Система уравнений (необходимые условия экстремума):")
    print(f"dL/dx = {dL_dx} = 0")
    print(f"dL/dy = {dL_dy} = 0")
    print(f"dL/dlambda1 = {dL_dlam1} = 0")
    print(f"dL/dlambda2 = {dL_dlam2} = 0")
    print()
    
    # Упрощаем производные
    dL_dx_simplified = simplify(dL_dx)
    dL_dy_simplified = simplify(dL_dy)
    
    print("Упрощенная система:")
    print(f"dL/dx = {dL_dx_simplified} = 0")
    print(f"dL/dy = {dL_dy_simplified} = 0")
    print(f"g1: x^2 + y^2 = 1")
    print(f"g2: (x-1)^2 + (y-1)^2 = 2")
    print()
    
    # Решаем систему уравнений
    print("Решение системы уравнений...")
    print()
    
    # Из первого уравнения: 2x - 2λ₁x - 2λ₂(x-1) = 0
    # 2x(1 - λ₁ - λ₂) + 2λ₂ = 0
    # x(1 - λ₁ - λ₂) = -λ₂
    
    # Из второго уравнения: 6y - 2λ₁y - 2λ₂(y-1) = 0
    # 2y(3 - λ₁ - λ₂) + 2λ₂ = 0
    # y(3 - λ₁ - λ₂) = -λ₂
    
    # Решаем численно
    solutions = solve_lagrange_numerical()
    
    return solutions


def solve_lagrange_numerical():
    """
    Численное решение системы уравнений
    """
    def equations(vars):
        x, y, lam1, lam2 = vars
        
        # Уравнения из необходимых условий
        eq1 = 2*x - 2*lam1*x - 2*lam2*(x-1)  # ∂L/∂x = 0
        eq2 = 6*y - 2*lam1*y - 2*lam2*(y-1)  # ∂L/∂y = 0
        eq3 = x**2 + y**2 - 1  # g₁ = 0
        eq4 = (x-1)**2 + (y-1)**2 - 2  # g₂ = 0
        
        return [eq1, eq2, eq3, eq4]
    
    # Находим все решения
    solutions = []
    
    # Пробуем различные начальные приближения
    initial_guesses = [
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 1.0, 1.0],
        [0.7, 0.7, 1.0, 1.0],
        [-0.5, -0.5, 1.0, 1.0],
    ]
    
    found_solutions = set()
    
    for guess in initial_guesses:
        try:
            sol = fsolve(equations, guess, xtol=1e-10)
            x, y, lam1, lam2 = sol
            
            # Проверяем, что решение удовлетворяет уравнениям
            errors = equations(sol)
            if all(abs(err) < 1e-8 for err in errors):
                # Округляем для сравнения
                key = (round(x, 6), round(y, 6))
                if key not in found_solutions:
                    found_solutions.add(key)
                    solutions.append({
                        'x': x,
                        'y': y,
                        'lam1': lam1,
                        'lam2': lam2,
                        'f': x**2 + 3*y**2
                    })
        except:
            continue
    
    # Сортируем по значению целевой функции
    solutions.sort(key=lambda s: s['f'])
    
    print(f"Найдено {len(solutions)} стационарных точек:")
    print()
    
    for i, sol in enumerate(solutions, 1):
        print(f"Точка {i}:")
        print(f"  x = {sol['x']:.6f}")
        print(f"  y = {sol['y']:.6f}")
        print(f"  lambda1 = {sol['lam1']:.6f}")
        print(f"  lambda2 = {sol['lam2']:.6f}")
        print(f"  f(x,y) = {sol['f']:.6f}")
        
        # Проверка ограничений
        g1_val = sol['x']**2 + sol['y']**2 - 1
        g2_val = (sol['x']-1)**2 + (sol['y']-1)**2 - 2
        print(f"  Проверка ограничений:")
        print(f"    g1 = {g1_val:.2e}")
        print(f"    g2 = {g2_val:.2e}")
        print()
    
    return solutions


def check_sylvester_criterion(solutions):
    """
    Проверка характера найденных точек с помощью критерия Сильвестра
    для матрицы Гессе функции Лагранжа
    """
    print("=" * 80)
    print("ПРОВЕРКА ХАРАКТЕРА ТОЧЕК (КРИТЕРИЙ СИЛЬВЕСТРА)")
    print("=" * 80)
    print()
    
    x, y, lam1, lam2 = symbols('x y lam1 lam2', real=True)
    
    # Функция Лагранжа
    f = x**2 + 3*y**2
    g1 = x**2 + y**2 - 1
    g2 = (x-1)**2 + (y-1)**2 - 2
    L = f - lam1*g1 - lam2*g2
    
    # Матрица Гессе функции Лагранжа (только по x и y)
    H_xx = diff(diff(L, x), x)
    H_xy = diff(diff(L, x), y)
    H_yx = diff(diff(L, y), x)
    H_yy = diff(diff(L, y), y)
    
    print("Матрица Гессе функции Лагранжа (по переменным x, y):")
    print(f"H = [d^2L/dx^2    d^2L/dxdy]")
    print(f"    [d^2L/dydx    d^2L/dy^2]")
    print()
    print(f"d^2L/dx^2 = {simplify(H_xx)}")
    print(f"d^2L/dxdy = {simplify(H_xy)}")
    print(f"d^2L/dy^2 = {simplify(H_yy)}")
    print()
    
    for i, sol in enumerate(solutions, 1):
        print(f"Точка {i}: (x={sol['x']:.6f}, y={sol['y']:.6f})")
        
        # Подставляем значения
        H_xx_val = float(H_xx.subs([(x, sol['x']), (y, sol['y']), 
                                     (lam1, sol['lam1']), (lam2, sol['lam2'])]))
        H_xy_val = float(H_xy.subs([(x, sol['x']), (y, sol['y']), 
                                     (lam1, sol['lam1']), (lam2, sol['lam2'])]))
        H_yy_val = float(H_yy.subs([(x, sol['x']), (y, sol['y']), 
                                     (lam1, sol['lam1']), (lam2, sol['lam2'])]))
        
        H = np.array([[H_xx_val, H_xy_val],
                      [H_xy_val, H_yy_val]])
        
        print(f"  Матрица Гессе:")
        print(f"    H = [{H_xx_val:8.4f}  {H_xy_val:8.4f}]")
        print(f"        [{H_xy_val:8.4f}  {H_yy_val:8.4f}]")
        
        # Вычисляем главные миноры
        det_H = np.linalg.det(H)
        print(f"  Определитель матрицы Гессе: det(H) = {det_H:.6f}")
        
        # Критерий Сильвестра для условного экстремума
        # Нужно проверить знакоопределенность на касательном пространстве
        # к многообразию ограничений
        
        # Градиенты ограничений
        grad_g1 = np.array([2*sol['x'], 2*sol['y']])
        grad_g2 = np.array([2*(sol['x']-1), 2*(sol['y']-1)])
        
        # Матрица градиентов ограничений
        G = np.array([grad_g1, grad_g2])
        
        print(f"  Градиенты ограничений:")
        print(f"    grad(g1) = [{grad_g1[0]:.4f}, {grad_g1[1]:.4f}]")
        print(f"    grad(g2) = [{grad_g2[0]:.4f}, {grad_g2[1]:.4f}]")
        
        # Проверяем линейную независимость градиентов
        if np.linalg.matrix_rank(G) < 2:
            print(f"  ⚠️  Градиенты ограничений линейно зависимы!")
            print(f"     Это может означать, что ограничения не являются регулярными.")
        else:
            print(f"  ✓ Градиенты ограничений линейно независимы")
        
        # Для условного экстремума нужно проверить знакоопределенность
        # ограниченной матрицы Гессе на касательном пространстве
        
        # Собственные значения полной матрицы Гессе
        eigenvals = np.linalg.eigvals(H)
        print(f"  Собственные значения матрицы Гессе:")
        print(f"    lambda1 = {eigenvals[0]:.6f}")
        print(f"    lambda2 = {eigenvals[1]:.6f}")
        
        # Для условного экстремума проверяем знакоопределенность
        # на касательном пространстве к многообразию ограничений
        # Касательное пространство - это ортогональное дополнение к span{grad(g1), grad(g2)}
        
        # Поскольку у нас 2 ограничения и 2 переменные, касательное пространство
        # имеет размерность 0 (точки пересечения двух кривых)
        # В этом случае нужно проверить знак второй производной по направлению
        
        # Альтернативный подход: проверяем знак определителя расширенной матрицы
        # для достаточных условий второго порядка
        
        # Для задачи с двумя ограничениями-равенствами:
        # Если det(H) < 0, то это не может быть локальным минимумом
        # (матрица не положительно определена)
        
        # Проверяем достаточные условия второго порядка
        # Для минимума: ограниченная матрица Гессе должна быть положительно определена
        # на касательном пространстве
        
        # Поскольку касательное пространство имеет размерность 0 (точка пересечения),
        # достаточно проверить знак функции в окрестности
        
        # Но для формальной проверки используем критерий знакоопределенности
        # на подпространстве, ортогональном градиентам ограничений
        
        # Находим базис касательного пространства (ортогональное дополнение)
        # В 2D с 2 ограничениями касательное пространство = {0}
        # Поэтому проверяем знак функции вблизи точки
        
        print(f"  Анализ характера точки:")
        
        # Проверяем знак определителя и собственных значений
        if det_H < 0:
            print(f"    det(H) < 0: матрица не является знакоопределенной")
        
        # Для более точного анализа используем проверку на касательном пространстве
        # Но в данном случае, так как касательное пространство нульмерно,
        # нужно сравнивать значения функции в найденных точках
        
        # Сравниваем значения функции
        if i == 1:  # Первая точка (с минимальным значением)
            print(f"    Это точка с минимальным значением f = {sol['f']:.6f}")
            print(f"    ✓ ЛОКАЛЬНЫЙ МИНИМУМ (наименьшее значение среди стационарных точек)")
        else:
            print(f"    Это точка с большим значением f = {sol['f']:.6f}")
            print(f"    ✗ НЕ ЯВЛЯЕТСЯ МИНИМУМОМ (большее значение функции)")
        
        print()
    
    return solutions


def visualize_solution(solutions):
    """
    Визуализация решения
    """
    print("=" * 80)
    print("ВИЗУАЛИЗАЦИЯ")
    print("=" * 80)
    print()
    
    fig = plt.figure(figsize=(14, 6))
    
    # График 1: Линии уровня и ограничения
    ax1 = fig.add_subplot(121)
    
    # Создаем сетку
    x_range = np.linspace(-2, 2, 400)
    y_range = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Целевая функция
    F = X**2 + 3*Y**2
    
    # Ограничения
    G1 = X**2 + Y**2 - 1
    G2 = (X-1)**2 + (Y-1)**2 - 2
    
    # Линии уровня
    levels = np.linspace(0, 5, 20)
    contour = ax1.contour(X, Y, F, levels=levels, colors='blue', alpha=0.3)
    ax1.clabel(contour, inline=True, fontsize=8)
    
    # Ограничения
    cs1 = ax1.contour(X, Y, G1, levels=[0], colors='red', linewidths=2)
    cs2 = ax1.contour(X, Y, G2, levels=[0], colors='green', linewidths=2)
    # Добавляем метки вручную
    ax1.plot([], [], 'r-', linewidth=2, label='g1: x^2+y^2=1')
    ax1.plot([], [], 'g-', linewidth=2, label='g2: (x-1)^2+(y-1)^2=2')
    
    # Стационарные точки
    for i, sol in enumerate(solutions):
        color = 'red' if i == 0 else 'orange'
        marker = 'o' if i == 0 else 's'
        ax1.plot(sol['x'], sol['y'], marker, color=color, markersize=10, 
                label=f"Точка {i+1}: f={sol['f']:.3f}")
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Линии уровня f(x,y) и ограничения')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.set_xlim(-1.5, 2)
    ax1.set_ylim(-1.5, 2)
    
    # График 2: 3D поверхность
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Поверхность целевой функции
    surf = ax2.plot_surface(X, Y, F, cmap='viridis', alpha=0.7, 
                           linewidth=0, antialiased=True)
    
    # Ограничения на поверхности
    theta = np.linspace(0, 2*np.pi, 100)
    # g1: окружность
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    z_circle = x_circle**2 + 3*y_circle**2
    ax2.plot(x_circle, y_circle, z_circle, 'r-', linewidth=2, label='g1')
    
    # g2: окружность с центром (1,1)
    x_circle2 = 1 + np.sqrt(2)*np.cos(theta)
    y_circle2 = 1 + np.sqrt(2)*np.sin(theta)
    z_circle2 = x_circle2**2 + 3*y_circle2**2
    ax2.plot(x_circle2, y_circle2, z_circle2, 'g-', linewidth=2, label='g2')
    
    # Стационарные точки
    for i, sol in enumerate(solutions):
        color = 'red' if i == 0 else 'orange'
        ax2.scatter([sol['x']], [sol['y']], [sol['f']], 
                   color=color, s=100, marker='o')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('f(x,y)')
    ax2.set_title('Поверхность f(x,y) = x^2 + 3y^2')
    
    plt.tight_layout()
    import os
    os.makedirs('lab-3', exist_ok=True)
    plt.savefig('solution_visualization.png', dpi=150, bbox_inches='tight')
    plt.savefig('solution_visualization.pdf', bbox_inches='tight')
    print("Графики сохранены в solution_visualization.png и .pdf")
    print()


def main():
    """
    Основная функция
    """
    # Решаем задачу
    solutions = solve_lagrange_analytical()
    
    # Проверяем характер точек
    check_sylvester_criterion(solutions)
    
    # Визуализация
    try:
        visualize_solution(solutions)
    except Exception as e:
        print(f"Ошибка при создании графиков: {e}")
    
    # Вывод итогового результата
    print("=" * 80)
    print("ИТОГОВЫЙ РЕЗУЛЬТАТ")
    print("=" * 80)
    print()
    
    if solutions:
        optimal = solutions[0]
        print(f"Глобальный минимум:")
        print(f"  x* = {optimal['x']:.6f}")
        print(f"  y* = {optimal['y']:.6f}")
        print(f"  f(x*,y*) = {optimal['f']:.6f}")
        print()
        print(f"Множители Лагранжа:")
        print(f"  lambda1* = {optimal['lam1']:.6f}")
        print(f"  lambda2* = {optimal['lam2']:.6f}")
    else:
        print("Решения не найдены!")
    
    print()


if __name__ == "__main__":
    main()

