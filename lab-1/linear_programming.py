import numpy as np

class LinearProgrammingSolver:
    def __init__(self):
        self.objective = []  # Коэффициенты целевой функции
        self.constraints = []  # Ограничения
        self.variables = []  # Имена переменных
        self.is_maximize = True  # Минимизация или максимизация
        self.solution = None  # Решение
        self.optimal_value = None  # Оптимальное значение
        
        # Промежуточные результаты для демонстрации
        self.canonical_form = None
        self.auxiliary_problem = None
    
    def parse_problem(self, filename: str):
        """Парсинг задачи из файла"""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                content = file.read().strip()
            
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            self.variables = []
            self.objective = []
            self.is_maximize = True
            self.constraints = []
            
            # Сначала парсим переменные, если они есть
            variables_section = False
            for line in lines:
                if line.upper().startswith('VARIABLES'):
                    variables_section = True
                    self._parse_variables(line)
                    break
            
            # Если переменные не найдены, извлекаем их из целевой функции
            if not self.variables:
                for line in lines:
                    if line.upper() in ['MAXIMIZE', 'MINIMIZE']:
                        continue
                    elif line.upper() == 'SUBJECT_TO':
                        break
                    elif line and not line.upper().startswith('VARIABLES'):
                        # Это должна быть целевая функция
                        self._extract_variables_from_objective(line)
                        break
            
            current_section = None
            
            for line in lines:
                line_upper = line.upper()
                
                # Определяем секцию
                if line_upper == 'MAXIMIZE':
                    current_section = 'objective'
                    self.is_maximize = True
                    continue
                elif line_upper == 'MINIMIZE':
                    current_section = 'objective'
                    self.is_maximize = False
                    continue
                elif line_upper == 'SUBJECT_TO':
                    current_section = 'constraints'
                    continue
                elif line_upper.startswith('VARIABLES'):
                    current_section = 'variables'
                    continue
                
                # Обработка секций
                if current_section == 'objective':
                    self._parse_objective(line)
                elif current_section == 'constraints':
                    self._parse_constraint(line)
                elif current_section == 'variables':
                    self._parse_variables(line)
            
            print(f"Задача успешно загружена из файла {filename}")
            print(f"Переменные: {self.variables}")
            print(f"Целевая функция: {self.objective}")
            print(f"Тип задачи: {'Максимизация' if self.is_maximize else 'Минимизация'}")
            print(f"Количество ограничений: {len(self.constraints)}")
            
        except FileNotFoundError:
            print(f"Ошибка: Файл {filename} не найден")
            raise
        except Exception as e:
            print(f"Ошибка при чтении файла: {e}")
            raise
    
    def _extract_variables_from_objective(self, line: str):
        """Извлечение переменных из целевой функции"""
        import re
        # Находим все переменные вида x1, x2, etc.
        variables = re.findall(r'x\d+', line)
        self.variables = sorted(list(set(variables)))
    
    def _parse_objective(self, line: str):
        """Парсинг целевой функции"""
        terms = line.replace(' ', '').split('+')
        self.objective = []
        
        for term in terms:
            term = term.strip()
            if '*' in term:
                coeff, var = term.split('*')
                self.objective.append(float(coeff))
            else:
                self.objective.append(float(term))
    
    def _parse_constraint(self, line: str):
        """Парсинг ограничения"""
        # Ищем знак сравнения
        signs = ['<=', '>=', '=']
        sign_found = None
        sign_pos = -1
        
        for sign in signs:
            pos = line.find(sign)
            if pos != -1:
                sign_found = sign
                sign_pos = pos
                break
        
        if sign_found is None:
            raise ValueError(f"Не найден знак сравнения в ограничении: {line}")
        
        # Разделяем на левую и правую части
        left_part = line[:sign_pos].strip()
        right_part = line[sign_pos + len(sign_found):].strip()
        
        # Парсим левую часть (коэффициенты)
        coefficients = self._parse_expression(left_part)
        
        # Убеждаемся, что у нас есть коэффициенты для всех переменных
        while len(coefficients) < len(self.variables):
            coefficients.append(0.0)
        
        # Парсим правую часть (число)
        rhs = float(right_part)
        
        self.constraints.append({
            'coefficients': coefficients,
            'sign': sign_found,
            'rhs': rhs
        })
    
    def _parse_variables(self, line: str):
        # Удаляем префикс "VARIABLES" если есть
        if line.upper().startswith('VARIABLES'):
            line = line[9:].strip()
        
        self.variables = line.split()
    
    def _parse_expression(self, expr: str):
        import re

        expr = expr.replace(' ', '')
        
        # Инициализируем массив коэффициентов для всех переменных
        coefficients = [0.0] * len(self.variables)
        
        # Паттерн для поиска слагаемых: коэффициент*переменная или просто переменная
        pattern = r'([+-]?\d*\.?\d*)\*?([a-zA-Z]\d*)'
        matches = re.findall(pattern, expr)
        
        for coeff_str, var in matches:
            if coeff_str == '' or coeff_str == '+':
                coeff = 1.0
            elif coeff_str == '-':
                coeff = -1.0
            else:
                coeff = float(coeff_str)
            
            # Находим индекс переменной
            try:
                var_idx = self.variables.index(var)
                coefficients[var_idx] = coeff
            except ValueError:
                # Если переменная не найдена, добавляем её
                self.variables.append(var)
                coefficients.append(coeff)
        
        return coefficients
    
    def to_canonical_form(self):
        """Приведение задачи к каноническому виду"""
        
        n_vars = len(self.variables)
        
        # Определяем, сколько slack переменных нужно
        num_slack = sum(1 for c in self.constraints if c['sign'] in ['<=', '>='])
        
        A_list = []
        b_list = []
        slack_idx = 0
        
        print("Исходные ограничения:")
        for i, constraint in enumerate(self.constraints):
            coeffs = constraint['coefficients'].copy()
            sign = constraint['sign']
            rhs = constraint['rhs']
            
            print(f"{i+1}. {coeffs} {sign} {rhs}")
            
            row = coeffs.copy()
            
            if sign == '<=':
                # Добавляем slack переменные: нули для предыдущих, 1 для текущей
                while len(row) < n_vars:
                    row.append(0)
                
                # Добавляем slack переменные
                for j in range(num_slack):
                    if j == slack_idx:
                        row.append(1)
                    else:
                        row.append(0)
                slack_idx += 1
                A_list.append(row)
                b_list.append(rhs)
                
            elif sign == '>=':
                # Вычитаем slack переменные
                while len(row) < n_vars:
                    row.append(0)
                
                # Добавляем slack переменные
                for j in range(num_slack):
                    if j == slack_idx:
                        row.append(-1)
                    else:
                        row.append(0)
                slack_idx += 1
                A_list.append(row)
                b_list.append(-rhs)  # Меняем знак правой части
                
            elif sign == '=':
                # Для равенства не добавляем slack
                while len(row) < n_vars:
                    row.append(0)
                
                # Добавляем нули для всех slack переменных
                for j in range(num_slack):
                    row.append(0)
                
                A_list.append(row)
                b_list.append(rhs)
        
        A = np.array(A_list, dtype=float)
        b = np.array(b_list, dtype=float)
        
        print(f"\nМатрица A (канонический вид):\n{A}")
        print(f"\nВектор b: {b}")
        
        self.canonical_form = {'A': A, 'b': b}
        return A, b
    
    def solve_with_simplex(self, use_manual=True):
        """Решение задачи симплекс-методом"""
        
        print("\nЭТАП 1: ПРИВЕДЕНИЕ К КАНОНИЧЕСКОМУ ВИДУ\n")
        A, b = self.to_canonical_form()
        
        if use_manual:
            return self._solve_manual_simplex(A, b)
        else:
            return self._solve_scipy_simplex()
    
    def _solve_scipy_simplex(self):
        """Решение с использованием scipy"""
        try:
            from scipy.optimize import linprog
            
            print("\nЭТАП 2: ВСПОМОГАТЕЛЬНАЯ ЗАДАЧА\n")
            print("Цель вспомогательной задачи: найти допустимое решение")
            print("Для ограничений типа '=' и '>=' добавляем искусственные переменные")
            
            # Если есть ограничения '=' или '>=', нужна искусственная переменная
            artificial_needed = any(c['sign'] in ['=', '>='] for c in self.constraints)
            
            if artificial_needed:
                print("Обнаружены ограничения, требующие искусственных переменных")
                print("В данной реализации используется стандартный метод linprog")
                print("который автоматически обрабатывает искусственные переменные.\n")
            

            print("\nЭТАП 3: РЕШЕНИЕ ОСНОВНОЙ ЗАДАЧИ\n")
            
            c = np.array(self.objective)
            
            A_eq = []
            b_eq = []
            A_ub = []
            b_ub = []
            
            for constraint in self.constraints:
                coeffs = np.array(constraint['coefficients'])
                
                if constraint['sign'] == '<=':
                    A_ub.append(coeffs)
                    b_ub.append(constraint['rhs'])
                elif constraint['sign'] == '>=':
                    A_ub.append(-coeffs)
                    b_ub.append(-constraint['rhs'])
                elif constraint['sign'] == '=':
                    A_eq.append(coeffs)
                    b_eq.append(constraint['rhs'])
            
            if A_ub:
                A_ub = np.array(A_ub)
                b_ub = np.array(b_ub)
            else:
                A_ub = None
                b_ub = None
            
            if A_eq:
                A_eq = np.array(A_eq)
                b_eq = np.array(b_eq)
            else:
                A_eq = None
                b_eq = None

            bounds = [(0, None)] * len(self.objective)
            
            # Если максимизация, меняем знак целевой функции
            if self.is_maximize:
                c = -c
            
            print(f"Целевая функция: {'Max' if self.is_maximize else 'Min'} {self.objective}")
            print(f"Количество ограничений: {len(self.constraints)}")
            
            # Решение основной задачи
            result = linprog(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method='highs'
            )
            
            if result.success:
                if self.is_maximize:
                    optimal_value = result.fun
                else:
                    optimal_value = -result.fun
                
                self.solution = result.x
                self.optimal_value = optimal_value
                
                print(f"\nРЕШЕНИЕ НАЙДЕНО")
                print(f"Оптимальная точка: {result.x}")
                print(f"Значение целевой функции: {optimal_value}")
                
                return result.x, optimal_value
            else:
                print(f"\nРЕШЕНИЕ НЕ НАЙДЕНО")
                print(f"Причина: {result.message}")
                
                return None, None
        except ImportError:
            print("Scipy не установлен, используем ручную реализацию")
            return self._solve_manual_simplex(None, None)
    
    def _solve_manual_simplex(self, A, b):
        """Ручная реализация симплекс-метода"""
        print("\nЭТАП 2: РУЧНАЯ РЕАЛИЗАЦИЯ СИМПЛЕКС-МЕТОДА\n")
        
        # Проверяем, есть ли искусственные переменные
        num_artificial = sum(1 for c in self.constraints if c['sign'] in ['=', '>='])
        
        if num_artificial > 0:
            print("Обнаружены искусственные переменные - используем двухфазный метод")
            return self._two_phase_simplex()
        else:
            print("Искусственных переменных нет - используем стандартный симплекс-метод")
            return self._standard_simplex()
    
    def _standard_simplex(self):
        tableau = self._create_simplex_tableau()
        
        print("Начальная симплекс-таблица:")
        self._print_tableau(tableau)
        
        iteration = 0
        max_iterations = 100
        
        while iteration < max_iterations:
            iteration += 1
            
            # Проверяем оптимальность
            if self._is_optimal(tableau):
                print(f"\nОптимальное решение найдено на итерации {iteration}")
                break
            
            # Выбираем входящую переменную (столбец)
            entering_col = self._find_entering_variable(tableau)
            if entering_col == -1:
                print("Не удалось найти входящую переменную")
                break
            
            # Выбираем выходящую переменную (строку)
            leaving_row = self._find_leaving_variable(tableau, entering_col)
            if leaving_row == -1:
                print("Задача неограничена")
                break
            
            print(f"\nИтерация {iteration}:")
            print(f"Входящая переменная: x{entering_col + 1}")
            print(f"Выходящая переменная: строка {leaving_row + 1}")
            
            # Выполняем операцию поворота
            tableau = self._pivot(tableau, leaving_row, entering_col)
            
            print(f"Таблица после поворота:")
            self._print_tableau(tableau)
        
        if iteration >= max_iterations:
            print("Достигнуто максимальное количество итераций")
            return None, None

        solution = self._extract_solution(tableau)
        optimal_value = tableau[0, -1]
        
        # Для максимизации значение в таблице отрицательное (мы меняли знак)
        if self.is_maximize:
            optimal_value = -optimal_value
        
        self.solution = solution
        self.optimal_value = optimal_value
        
        print(f"\nРЕШЕНИЕ НАЙДЕНО")
        print(f"Оптимальная точка: {solution}")
        print(f"Значение целевой функции: {optimal_value}")
        
        return solution, optimal_value
    
    def _two_phase_simplex(self):
        """Двухфазный симплекс-метод для задач с искусственными переменными"""
        print("\nФАЗА 1: Минимизация суммы искусственных переменных")
        
        # Создаем таблицу для фазы 1
        tableau = self._create_phase1_tableau()
        
        print("Начальная таблица фазы 1:")
        self._print_tableau(tableau)
        
        # Решаем фазу 1
        iteration = 0
        max_iterations = 100
        
        while iteration < max_iterations:
            iteration += 1
            
            # Проверяем оптимальность для фазы 1
            if self._is_phase1_optimal(tableau):
                print(f"\nФаза 1 завершена на итерации {iteration}")
                break
            
            # Выбираем входящую переменную
            entering_col = self._find_phase1_entering_variable(tableau)
            if entering_col == -1:
                print("Не удалось найти входящую переменную в фазе 1")
                break
            
            # Выбираем выходящую переменную
            leaving_row = self._find_leaving_variable(tableau, entering_col)
            if leaving_row == -1:
                print("Задача неограничена в фазе 1")
                break
            
            print(f"\nФаза 1, итерация {iteration}:")
            print(f"Входящая переменная: x{entering_col + 1}")
            print(f"Выходящая переменная: строка {leaving_row + 1}")
            
            # Выполняем операцию поворота
            tableau = self._pivot(tableau, leaving_row, entering_col)
            
            print(f"Таблица после поворота:")
            self._print_tableau(tableau)
        
        # Проверяем результат фазы 1
        phase1_value = tableau[0, -1]
        if abs(phase1_value) > 1e-10:
            print(f"\nФаза 1: значение искусственных переменных = {phase1_value}")
            print("Задача несовместна - нет допустимого решения")
            return None, None
        
        print("\nФАЗА 2: Решение исходной задачи")
        
        tableau = self._create_phase2_tableau(tableau)
        
        print("Начальная таблица фазы 2:")
        self._print_tableau(tableau)
        
        # Решаем фазу 2
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Проверяем оптимальность для фазы 2
            if self._is_optimal(tableau):
                print(f"\nОптимальное решение найдено на итерации {iteration}")
                break
            
            # Выбираем входящую переменную
            entering_col = self._find_entering_variable(tableau)
            if entering_col == -1:
                print("Не удалось найти входящую переменную в фазе 2")
                break
            
            # Выбираем выходящую переменную
            leaving_row = self._find_leaving_variable(tableau, entering_col)
            if leaving_row == -1:
                print("Задача неограничена в фазе 2")
                break
            
            print(f"\nФаза 2, итерация {iteration}:")
            print(f"Входящая переменная: x{entering_col + 1}")
            print(f"Выходящая переменная: строка {leaving_row + 1}")
            
            # Выполняем операцию поворота
            tableau = self._pivot(tableau, leaving_row, entering_col)
            
            print(f"Таблица после поворота:")
            self._print_tableau(tableau)
        
        if iteration >= max_iterations:
            print("Достигнуто максимальное количество итераций в фазе 2")
            return None, None
        
        solution = self._extract_solution(tableau)
        optimal_value = tableau[0, -1]
        
        # Для максимизации значение в таблице отрицательное
        if self.is_maximize:
            optimal_value = -optimal_value
        
        self.solution = solution
        self.optimal_value = optimal_value
        
        print(f"\nРЕШЕНИЕ НАЙДЕНО")
        print(f"Оптимальная точка: {solution}")
        print(f"Значение целевой функции: {optimal_value}")
        
        return solution, optimal_value
    
    def _create_simplex_tableau(self):
        n_vars = len(self.variables)
        n_constraints = len(self.constraints)
        
        # Подсчитываем количество slack и искусственных переменных
        num_slack = sum(1 for c in self.constraints if c['sign'] in ['<=', '>='])
        num_artificial = sum(1 for c in self.constraints if c['sign'] in ['=', '>='])
        
        # Размер таблицы: (n_constraints + 1) x (n_vars + num_slack + num_artificial + 1)
        tableau = np.zeros((n_constraints + 1, n_vars + num_slack + num_artificial + 1))
        
        # Заполняем целевую функцию (первая строка)
        for i in range(n_vars):
            if self.is_maximize:
                tableau[0, i] = -self.objective[i]  # Для максимизации меняем знак
            else:
                tableau[0, i] = self.objective[i]
        
        # Заполняем ограничения
        slack_idx = 0
        artificial_idx = 0
        
        for row_idx, constraint in enumerate(self.constraints):
            row = row_idx + 1
            
            # Коэффициенты переменных
            for i in range(n_vars):
                tableau[row, i] = constraint['coefficients'][i]
            
            # Slack и искусственные переменные
            if constraint['sign'] == '<=':
                tableau[row, n_vars + slack_idx] = 1
                tableau[row, -1] = constraint['rhs']
                slack_idx += 1
            elif constraint['sign'] == '>=':
                tableau[row, n_vars + slack_idx] = -1
                tableau[row, n_vars + num_slack + artificial_idx] = 1
                tableau[row, -1] = -constraint['rhs']
                slack_idx += 1
                artificial_idx += 1
            elif constraint['sign'] == '=':
                tableau[row, n_vars + num_slack + artificial_idx] = 1
                tableau[row, -1] = constraint['rhs']
                artificial_idx += 1
        
        # Если есть искусственные переменные, используем двухфазный метод
        if num_artificial > 0:
            # Фаза 1: минимизируем сумму искусственных переменных
            for i in range(num_artificial):
                tableau[0, n_vars + num_slack + i] = 1
            
            # Вычитаем строки с искусственными переменными из строки Z
            for row_idx, constraint in enumerate(self.constraints):
                if constraint['sign'] in ['=', '>=']:
                    row = row_idx + 1
                    tableau[0, :] -= tableau[row, :]
        
        return tableau
    
    def _print_tableau(self, tableau):
        """Печать симплекс-таблицы"""
        n_vars = len(self.variables)
        n_rows, n_cols = tableau.shape
        
        # Заголовки столбцов
        headers = [f"x{i+1}" for i in range(n_vars)]
        
        # Подсчитываем количество slack и искусственных переменных
        num_slack = sum(1 for c in self.constraints if c['sign'] in ['<=', '>='])
        num_artificial = sum(1 for c in self.constraints if c['sign'] in ['=', '>='])
        
        headers.extend([f"s{i+1}" for i in range(num_slack)])
        headers.extend([f"a{i+1}" for i in range(num_artificial)])
        headers.append("RHS")

        print("     " + "".join([f"{h:>8}" for h in headers]))
        
        for i in range(n_rows):
            if i == 0:
                print("Z   ", end="")
            else:
                print(f"s{i}  ", end="")
            
            for j in range(n_cols):
                print(f"{tableau[i, j]:>8.2f}", end="")
            print()
    
    def _is_optimal(self, tableau):
        """Проверка оптимальности"""
        # Для максимизации: все коэффициенты в строке Z должны быть >= 0
        # Для минимизации: все коэффициенты в строке Z должны быть <= 0
        z_row = tableau[0, :-1]  # Исключаем столбец RHS
        
        if self.is_maximize:
            return np.all(z_row >= -1e-10)  # Небольшая погрешность для численных вычислений
        else:
            return np.all(z_row <= 1e-10)
    
    def _find_entering_variable(self, tableau):
        """Поиск входящей переменной"""
        z_row = tableau[0, :-1]  # Исключаем столбец RHS
        
        if self.is_maximize:
            # Для максимизации: выбираем переменную с наименьшим отрицательным коэффициентом
            min_coeff = np.min(z_row)
            if min_coeff >= -1e-10:
                return -1  # Оптимально
            return np.argmin(z_row)
        else:
            # Для минимизации: выбираем переменную с наибольшим положительным коэффициентом
            max_coeff = np.max(z_row)
            if max_coeff <= 1e-10:
                return -1  # Оптимально
            return np.argmax(z_row)
    
    def _find_leaving_variable(self, tableau, entering_col):
        """Поиск выходящей переменной"""
        rhs_col = tableau[:, -1]
        entering_col_values = tableau[1:, entering_col]  # Исключаем строку Z
        
        # Находим минимальное положительное отношение RHS / entering_col
        ratios = []
        for i in range(len(entering_col_values)):
            if entering_col_values[i] > 1e-10:  # Избегаем деления на ноль
                ratio = rhs_col[i + 1] / entering_col_values[i]  # +1 потому что пропускаем строку Z
                ratios.append(ratio)
            else:
                ratios.append(float('inf'))
        
        if not ratios or all(r == float('inf') for r in ratios):
            return -1  # Задача неограничена
        
        min_ratio_idx = np.argmin(ratios)
        return min_ratio_idx
    
    def _pivot(self, tableau, leaving_row, entering_col):
        """Операция поворота"""
        tableau = tableau.copy()
        
        # Нормализуем ведущую строку
        pivot_element = tableau[leaving_row + 1, entering_col]  # +1 потому что пропускаем строку Z
        tableau[leaving_row + 1, :] /= pivot_element
        
        # Обновляем остальные строки
        for i in range(tableau.shape[0]):
            if i != leaving_row + 1:  # Пропускаем ведущую строку
                factor = tableau[i, entering_col]
                tableau[i, :] -= factor * tableau[leaving_row + 1, :]
        
        return tableau
    
    def _extract_solution(self, tableau):
        n_vars = len(self.variables)
        solution = np.zeros(n_vars)
        
        # Находим базисные переменные
        for col in range(n_vars):
            # Ищем столбец с одной единицей и остальными нулями
            col_values = tableau[1:, col]  # Исключаем строку Z
            
            # Проверяем, является ли переменная базисной
            unit_rows = np.where(np.abs(col_values - 1) < 1e-10)[0]
            zero_rows = np.where(np.abs(col_values) < 1e-10)[0]
            
            # Переменная базисная, если есть ровно одна единица и остальные нули
            if len(unit_rows) == 1 and len(unit_rows) + len(zero_rows) == len(col_values):
                row_idx = unit_rows[0]
                solution[col] = tableau[row_idx + 1, -1]  # Значение из столбца RHS
            else:
                # Переменная не в базисе, значение = 0
                solution[col] = 0.0
        
        return solution
    
    def _create_phase1_tableau(self):
        """Создание таблицы для фазы 1 (минимизация искусственных переменных)"""
        n_vars = len(self.variables)
        n_constraints = len(self.constraints)
        
        # Подсчитываем количество slack и искусственных переменных
        num_slack = sum(1 for c in self.constraints if c['sign'] in ['<=', '>='])
        num_artificial = sum(1 for c in self.constraints if c['sign'] in ['=', '>='])
        
        # Размер таблицы: (n_constraints + 1) x (n_vars + num_slack + num_artificial + 1)
        tableau = np.zeros((n_constraints + 1, n_vars + num_slack + num_artificial + 1))
        
        # Заполняем целевую функцию фазы 1 (минимизация искусственных переменных)
        # Все коэффициенты исходной целевой функции = 0
        # Коэффициенты искусственных переменных = 1
        
        # Заполняем ограничения
        slack_idx = 0
        artificial_idx = 0
        
        for row_idx, constraint in enumerate(self.constraints):
            row = row_idx + 1
            
            # Коэффициенты переменных
            for i in range(n_vars):
                tableau[row, i] = constraint['coefficients'][i]
            
            # Slack и искусственные переменные
            if constraint['sign'] == '<=':
                tableau[row, n_vars + slack_idx] = 1
                tableau[row, -1] = constraint['rhs']
                slack_idx += 1
            elif constraint['sign'] == '>=':
                tableau[row, n_vars + slack_idx] = -1
                tableau[row, n_vars + num_slack + artificial_idx] = 1
                tableau[row, -1] = -constraint['rhs']
                slack_idx += 1
                artificial_idx += 1
            elif constraint['sign'] == '=':
                tableau[row, n_vars + num_slack + artificial_idx] = 1
                tableau[row, -1] = constraint['rhs']
                artificial_idx += 1
        
        # Устанавливаем коэффициенты искусственных переменных в строке Z = 1
        for i in range(num_artificial):
            tableau[0, n_vars + num_slack + i] = 1
        
        # Вычитаем строки с искусственными переменными из строки Z
        for row_idx, constraint in enumerate(self.constraints):
            if constraint['sign'] in ['=', '>=']:
                row = row_idx + 1
                tableau[0, :] -= tableau[row, :]
        
        return tableau
    
    def _create_phase2_tableau(self, phase1_tableau):
        """Создание таблицы для фазы 2 (решение исходной задачи)"""
        n_vars = len(self.variables)
        n_constraints = len(self.constraints)
        
        # Подсчитываем количество slack и искусственных переменных
        num_slack = sum(1 for c in self.constraints if c['sign'] in ['<=', '>='])
        num_artificial = sum(1 for c in self.constraints if c['sign'] in ['=', '>='])
        
        # Создаем новую таблицу без искусственных переменных
        tableau = np.zeros((n_constraints + 1, n_vars + num_slack + 1))
        
        # Копируем ограничения из фазы 1 (без искусственных переменных)
        for i in range(n_constraints + 1):
            for j in range(n_vars):
                tableau[i, j] = phase1_tableau[i, j]
            for j in range(num_slack):
                tableau[i, n_vars + j] = phase1_tableau[i, n_vars + j]
            tableau[i, -1] = phase1_tableau[i, -1]
        
        # Устанавливаем целевую функцию исходной задачи
        for i in range(n_vars):
            if self.is_maximize:
                tableau[0, i] = -self.objective[i]  # Для максимизации меняем знак
            else:
                tableau[0, i] = self.objective[i]
        
        # Обнуляем коэффициенты slack переменных в строке Z
        for i in range(num_slack):
            tableau[0, n_vars + i] = 0
        
        # Вычитаем строки с базисными переменными из строки Z
        for col in range(n_vars):
            col_values = tableau[1:, col]
            unit_rows = np.where(np.abs(col_values - 1) < 1e-10)[0]
            
            if len(unit_rows) == 1:
                row_idx = unit_rows[0]
                coeff = tableau[0, col]
                tableau[0, :] -= coeff * tableau[row_idx + 1, :]
        
        return tableau
    
    def _is_phase1_optimal(self, tableau):
        """Проверка оптимальности для фазы 1"""
        # В фазе 1 минимизируем сумму искусственных переменных
        # Оптимально, когда все коэффициенты в строке Z >= 0
        z_row = tableau[0, :-1]  # Исключаем столбец RHS
        return np.all(z_row >= -1e-10)
    
    def _find_phase1_entering_variable(self, tableau):
        """Поиск входящей переменной для фазы 1"""
        z_row = tableau[0, :-1]  # Исключаем столбец RHS
        
        # Для минимизации выбираем переменную с наименьшим отрицательным коэффициентом
        min_coeff = np.min(z_row)
        if min_coeff >= -1e-10:
            return -1  # Оптимально
        return np.argmin(z_row)
    
    def format_solution(self):
        if self.solution is None:
            return "Решение не найдено"
        
        solution_str = "Оптимальная точка:\n"
        for i, (var, val) in enumerate(zip(self.variables, self.solution[:len(self.variables)])):
            solution_str += f"  {var} = {val:.4f}\n"
        
        solution_str += f"\nЗначение целевой функции: {self.optimal_value:.4f}"
        
        return solution_str


if __name__ == "__main__":
    solver = LinearProgrammingSolver()
    
    solver.parse_problem('problem.txt')
    
    print("\nЦелевая функция:")
    obj_str = " + ".join([f"{c}*{v}" for c, v in zip(solver.objective, solver.variables)])
    print(f"  Z = {obj_str}")
    
    print("\nОграничения:")
    for i, constraint in enumerate(solver.constraints):
        print(f"  {i+1}. {constraint['coefficients']} {constraint['sign']} {constraint['rhs']}")
  
    use_manual = True  # True для ручной реализации, False для scipy
    
    solution, opt_value = solver.solve_with_simplex(use_manual=use_manual)
    
    print("ИТОГОВОЕ РЕШЕНИЕ")
    print(solver.format_solution())
