import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum


class Situation(Enum):
    """Типы ситуаций на рынке"""
    FAVORABLE = "благоприятная"
    NEUTRAL = "нейтральная"
    NEGATIVE = "негативная"


@dataclass
class State:
    """Состояние портфеля"""
    cb1: float  # ЦБ1
    cb2: float  # ЦБ2
    deposit: float  # Депозиты
    free_cash: float  # Свободные средства
    
    def __hash__(self):
        # Более агрессивная дискретизация для ускорения вычислений
        # Округляем до 50 для ЦБ1, до 100 для ЦБ2, до 50 для депозитов, до 100 для свободных средств
        return hash((round(self.cb1 / 50) * 50, round(self.cb2 / 100) * 100, 
                    round(self.deposit / 50) * 50, round(self.free_cash / 100) * 100))
    
    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        # Используем более грубое сравнение для ускорения (дискретизация)
        return (abs(round(self.cb1 / 50) * 50 - round(other.cb1 / 50) * 50) < 1 and
                abs(round(self.cb2 / 100) * 100 - round(other.cb2 / 100) * 100) < 1 and
                abs(round(self.deposit / 50) * 50 - round(other.deposit / 50) * 50) < 1 and
                abs(round(self.free_cash / 100) * 100 - round(other.free_cash / 100) * 100) < 1)
    
    def total_value(self) -> float:
        """Общая стоимость портфеля"""
        return self.cb1 + self.cb2 + self.deposit + self.free_cash


@dataclass
class Action:
    """Действие управления портфелем"""
    delta_cb1: int  # Изменение ЦБ1 (в единицах шага: -1, 0, 1, ...)
    delta_cb2: int  # Изменение ЦБ2
    delta_deposit: int  # Изменение депозитов
    
    def cost(self, step_size_cb1: float, step_size_cb2: float, step_size_deposit: float,
             commission_cb1: float = 0, commission_cb2: float = 0, commission_deposit: float = 0) -> float:
        """Стоимость действия с учетом комиссий (положительная - покупка, отрицательная - продажа)"""
        cost_cb1 = self.delta_cb1 * step_size_cb1
        cost_cb2 = self.delta_cb2 * step_size_cb2
        cost_deposit = self.delta_deposit * step_size_deposit
        
        # Добавляем комиссию при покупке (положительное изменение)
        if cost_cb1 > 0:
            cost_cb1 *= (1 + commission_cb1)
        if cost_cb2 > 0:
            cost_cb2 *= (1 + commission_cb2)
        if cost_deposit > 0:
            cost_deposit *= (1 + commission_deposit)
        
        # При продаже вычитаем комиссию (отрицательное изменение)
        if cost_cb1 < 0:
            cost_cb1 *= (1 - commission_cb1)
        if cost_cb2 < 0:
            cost_cb2 *= (1 - commission_cb2)
        if cost_deposit < 0:
            cost_deposit *= (1 - commission_deposit)
        
        return cost_cb1 + cost_cb2 + cost_deposit
    
    def __eq__(self, other):
        if not isinstance(other, Action):
            return False
        return (self.delta_cb1 == other.delta_cb1 and
                self.delta_cb2 == other.delta_cb2 and
                self.delta_deposit == other.delta_deposit)
    
    def __hash__(self):
        return hash((self.delta_cb1, self.delta_cb2, self.delta_deposit))


class InvestmentPortfolioOptimizer:
    """Оптимизатор инвестиционного портфеля методом динамического программирования"""
    
    def __init__(self):
        # Начальное состояние
        self.initial_state = State(
            cb1=100.0,
            cb2=800.0,
            deposit=400.0,
            free_cash=600.0
        )
        
        # Размеры шагов управления (1/4 от первоначальной стоимости)
        self.step_size_cb1 = 25.0  # 100 / 4
        self.step_size_cb2 = 200.0  # 800 / 4
        self.step_size_deposit = 100.0  # 400 / 4
        
        # Минимальные ограничения (из CSV)
        self.min_cb1 = 30.0
        self.min_cb2 = 150.0
        self.min_deposit = 100.0
        
        # Комиссии брокеров (из CSV)
        self.commission_cb1 = 0.04  # 4%
        self.commission_cb2 = 0.07  # 7%
        self.commission_deposit = 0.05  # 5%
        
        # Данные по этапам: (вероятность, коэффициент_ЦБ1, коэффициент_ЦБ2, коэффициент_Деп)
        self.stage_data = [
            # Этап 1
            {
                Situation.FAVORABLE: (0.6, 1.20, 1.10, 1.07),
                Situation.NEUTRAL: (0.3, 1.05, 1.02, 1.03),
                Situation.NEGATIVE: (0.1, 0.80, 0.95, 1.00)
            },
            # Этап 2
            {
                Situation.FAVORABLE: (0.3, 1.4, 1.15, 1.01),
                Situation.NEUTRAL: (0.2, 1.05, 1.00, 1.00),
                Situation.NEGATIVE: (0.5, 0.60, 0.90, 1.00)
            },
            # Этап 3
            {
                Situation.FAVORABLE: (0.4, 1.15, 1.12, 1.05),
                Situation.NEUTRAL: (0.4, 1.05, 1.01, 1.01),
                Situation.NEGATIVE: (0.2, 0.70, 0.94, 1.00)
            }
        ]
        
        # Кэш для результатов динамического программирования
        self.dp_cache: Dict[Tuple[int, State], Tuple[float, Optional[Action]]] = {}
        
        # Ограничения на действия (максимальное количество шагов)
        # Уменьшено для ускорения вычислений, но достаточно для поиска оптимального решения
        self.max_steps = 3  # Максимальное количество шагов в каждом направлении
        
        # Счетчик вызовов для отладки
        self.recursion_count = 0
        
    def get_all_possible_actions(self, state: State) -> List[Action]:
        """Получить все возможные действия из текущего состояния"""
        actions = []
        
        # Определяем ограничения на действия
        # Можно продать максимум то, что есть, но не меньше минимального значения
        max_sell_cb1 = int((state.cb1 - self.min_cb1) / self.step_size_cb1)
        max_sell_cb2 = int((state.cb2 - self.min_cb2) / self.step_size_cb2)
        max_sell_deposit = int((state.deposit - self.min_deposit) / self.step_size_deposit)
        
        # Можно купить максимум на свободные средства
        max_buy_cb1 = int(state.free_cash / self.step_size_cb1)
        max_buy_cb2 = int(state.free_cash / self.step_size_cb2)
        max_buy_deposit = int(state.free_cash / self.step_size_deposit)
        
        # Ограничиваем диапазон для ускорения вычислений
        # Используем более разумные ограничения
        range_cb1 = range(-min(max_sell_cb1, self.max_steps), 
                         min(max_buy_cb1, self.max_steps) + 1)
        range_cb2 = range(-min(max_sell_cb2, self.max_steps), 
                         min(max_buy_cb2, self.max_steps) + 1)
        range_deposit = range(-min(max_sell_deposit, self.max_steps), 
                              min(max_buy_deposit, self.max_steps) + 1)
        
        # Генерируем все возможные комбинации действий
        for delta_cb1 in range_cb1:
            for delta_cb2 in range_cb2:
                for delta_deposit in range_deposit:
                    action = Action(delta_cb1, delta_cb2, delta_deposit)
                    
                    # Вычисляем стоимость с учетом комиссий
                    cost = action.cost(self.step_size_cb1, self.step_size_cb2, self.step_size_deposit,
                                      self.commission_cb1, self.commission_cb2, self.commission_deposit)
                    
                    # Проверяем допустимость действия (стоимость не превышает свободные средства)
                    if cost > state.free_cash + 1e-6:
                        continue
                    
                    # Проверяем, что не продаем больше, чем есть
                    if (delta_cb1 * self.step_size_cb1 < -state.cb1 - 1e-6 or
                        delta_cb2 * self.step_size_cb2 < -state.cb2 - 1e-6 or
                        delta_deposit * self.step_size_deposit < -state.deposit - 1e-6):
                        continue
                    
                    # Применяем действие для проверки минимальных ограничений
                    new_cb1 = state.cb1 + delta_cb1 * self.step_size_cb1
                    new_cb2 = state.cb2 + delta_cb2 * self.step_size_cb2
                    new_deposit = state.deposit + delta_deposit * self.step_size_deposit
                    
                    # Проверяем минимальные ограничения
                    if (new_cb1 < self.min_cb1 - 1e-6 or
                        new_cb2 < self.min_cb2 - 1e-6 or
                        new_deposit < self.min_deposit - 1e-6):
                        continue
                    
                    actions.append(action)
        
        # Всегда добавляем действие "ничего не делать" (если его еще нет)
        has_no_action = any(a.delta_cb1 == 0 and a.delta_cb2 == 0 and a.delta_deposit == 0 
                           for a in actions)
        if not has_no_action:
            actions.append(Action(0, 0, 0))
        
        return actions
    
    def apply_action(self, state: State, action: Action) -> State:
        """Применить действие к состоянию с учетом комиссий"""
        new_cb1 = state.cb1 + action.delta_cb1 * self.step_size_cb1
        new_cb2 = state.cb2 + action.delta_cb2 * self.step_size_cb2
        new_deposit = state.deposit + action.delta_deposit * self.step_size_deposit
        
        # Вычисляем стоимость с учетом комиссий
        cost = action.cost(self.step_size_cb1, self.step_size_cb2, self.step_size_deposit,
                          self.commission_cb1, self.commission_cb2, self.commission_deposit)
        new_free_cash = state.free_cash - cost
        
        # Обеспечиваем неотрицательность и минимальные ограничения
        new_cb1 = max(self.min_cb1, new_cb1)
        new_cb2 = max(self.min_cb2, new_cb2)
        new_deposit = max(self.min_deposit, new_deposit)
        new_free_cash = max(0, new_free_cash)
        
        return State(new_cb1, new_cb2, new_deposit, new_free_cash)
    
    def apply_situation(self, state: State, stage: int, situation: Situation) -> State:
        """Применить ситуацию к состоянию (изменение стоимости активов)"""
        prob, k_cb1, k_cb2, k_dep = self.stage_data[stage][situation]
        
        new_cb1 = state.cb1 * k_cb1
        new_cb2 = state.cb2 * k_cb2
        new_deposit = state.deposit * k_dep
        # Свободные средства не изменяются при изменении стоимости активов
        
        return State(new_cb1, new_cb2, new_deposit, state.free_cash)
    
    def bellman_recursion(self, stage: int, state: State) -> Tuple[float, Optional[Action]]:
        """
        Рекуррентное соотношение Беллмана
        
        F_k(s) = max_u [E[F_{k+1}(f(s, u, w))]]
        
        где:
        - k - номер этапа
        - s - состояние
        - u - управление (действие)
        - w - случайная ситуация
        - f(s, u, w) - функция перехода состояния
        """
        self.recursion_count += 1
        # if self.recursion_count % 10000 == 0:
        #     print(f"  Обработано {self.recursion_count} состояний...", flush=True)
        
        # Базовый случай: последний этап
        if stage >= len(self.stage_data):
            # Возвращаем общую стоимость портфеля как доход
            return state.total_value(), None
        
        # Проверяем кэш
        cache_key = (stage, state)
        if cache_key in self.dp_cache:
            return self.dp_cache[cache_key]
        
        best_value = -np.inf
        best_action = None
        
        # Получаем все возможные действия
        actions = self.get_all_possible_actions(state)
        
        # Для каждого действия вычисляем ожидаемое значение
        for action in actions:
            # Применяем действие
            new_state = self.apply_action(state, action)
            
            # Вычисляем ожидаемое значение после применения всех возможных ситуаций
            expected_value = 0.0
            
            for situation in Situation:
                prob, _, _, _ = self.stage_data[stage][situation]
                # Применяем ситуацию
                state_after_situation = self.apply_situation(new_state, stage, situation)
                # Рекурсивно вычисляем оптимальное значение для следующего этапа
                future_value, _ = self.bellman_recursion(stage + 1, state_after_situation)
                expected_value += prob * future_value
            
            # Обновляем лучшее значение
            if expected_value > best_value:
                best_value = expected_value
                best_action = action
        
        # Сохраняем в кэш
        self.dp_cache[cache_key] = (best_value, best_action)
        
        return best_value, best_action
    
    def solve(self) -> Tuple[float, List[Tuple[Action, State]]]:
        """
        Решение задачи динамического программирования
        
        Returns:
            (максимальный ожидаемый доход, список оптимальных действий и состояний)
        """
        self.dp_cache.clear()
        self.recursion_count = 0
        
        optimal_path = []
        current_state = self.initial_state
        
        # Проходим по всем этапам
        for stage in range(len(self.stage_data)):
            print(f"Этап {stage + 1}: поиск оптимального действия...")
            # Находим оптимальное действие для текущего этапа
            optimal_value, optimal_action = self.bellman_recursion(stage, current_state)
            
            if optimal_action is None:
                print(f"  Оптимальное действие не найдено")
                break
            
            print(f"  Найдено оптимальное действие для этапа {stage + 1}")
            
            # Применяем действие
            new_state = self.apply_action(current_state, optimal_action)
            optimal_path.append((optimal_action, new_state))
            
            # Применяем случайную ситуацию (для демонстрации используем ожидаемое значение)
            # В реальности ситуация выбирается случайно
            # Здесь мы используем критерий Байеса (ожидаемое значение)
            expected_state = State(0, 0, 0, 0)
            for situation in Situation:
                prob, _, _, _ = self.stage_data[stage][situation]
                state_after_situation = self.apply_situation(new_state, stage, situation)
                expected_state.cb1 += prob * state_after_situation.cb1
                expected_state.cb2 += prob * state_after_situation.cb2
                expected_state.deposit += prob * state_after_situation.deposit
                expected_state.free_cash += prob * state_after_situation.free_cash
            
            current_state = expected_state
            print(f"  Ожидаемое состояние после этапа {stage + 1}: стоимость = {current_state.total_value():.2f} д.е.")
            print()
        
        # Финальное значение
        final_value = current_state.total_value()
        
        print(f"Всего обработано состояний: {self.recursion_count}")
        print(f"Размер кэша: {len(self.dp_cache)}")
        print()
        
        return final_value, optimal_path
    
    def solve_with_trace(self) -> Dict:
        """
        Решение с детальной трассировкой для всех возможных сценариев
        """
        self.dp_cache.clear()
        
        # Находим оптимальное решение
        optimal_value, optimal_path = self.solve()
        
        # Строим дерево решений для всех возможных сценариев
        scenarios = []
        self._build_scenario_tree(0, self.initial_state, [], scenarios)
        
        return {
            'optimal_expected_value': optimal_value,
            'optimal_path': optimal_path,
            'scenarios': scenarios
        }
    
    def _build_scenario_tree(self, stage: int, state: State, path: List, scenarios: List):
        """Построение дерева всех возможных сценариев"""
        if stage >= len(self.stage_data):
            scenarios.append({
                'path': path.copy(),
                'final_state': state,
                'final_value': state.total_value()
            })
            return
        
        # Находим оптимальное действие
        _, optimal_action = self.bellman_recursion(stage, state)
        
        if optimal_action is None:
            return
        
        # Применяем действие
        new_state = self.apply_action(state, optimal_action)
        
        # Для каждой возможной ситуации
        for situation in Situation:
            prob, _, _, _ = self.stage_data[stage][situation]
            state_after_situation = self.apply_situation(new_state, stage, situation)
            
            path_entry = {
                'stage': stage,
                'action': optimal_action,
                'situation': situation,
                'probability': prob,
                'state_before': state,
                'state_after_action': new_state,
                'state_after_situation': state_after_situation
            }
            
            path.append(path_entry)
            self._build_scenario_tree(stage + 1, state_after_situation, path, scenarios)
            path.pop()


def main():
    optimizer = InvestmentPortfolioOptimizer()
    
    print("Начальное состояние портфеля:")
    initial = optimizer.initial_state
    print(f"  ЦБ1: {initial.cb1} д.е.")
    print(f"  ЦБ2: {initial.cb2} д.е.")
    print(f"  Депозиты: {initial.deposit} д.е.")
    print(f"  Свободные средства: {initial.free_cash} д.е.")
    print(f"  Общая стоимость: {initial.total_value()} д.е.")
    print()
    
    print("Размеры шагов управления:")
    print(f"  ЦБ1: {optimizer.step_size_cb1} д.е. (1/4 от {initial.cb1})")
    print(f"  ЦБ2: {optimizer.step_size_cb2} д.е. (1/4 от {initial.cb2})")
    print(f"  Депозиты: {optimizer.step_size_deposit} д.е. (1/4 от {initial.deposit})")
    print()
    
    print("Минимальные ограничения на активы:")
    print(f"  ЦБ1 >= {optimizer.min_cb1} д.е.")
    print(f"  ЦБ2 >= {optimizer.min_cb2} д.е.")
    print(f"  Депозиты >= {optimizer.min_deposit} д.е.")
    print()
    
    print("Комиссии брокеров:")
    print(f"  ЦБ1: {optimizer.commission_cb1*100:.0f}% (при покупке и продаже)")
    print(f"  ЦБ2: {optimizer.commission_cb2*100:.0f}% (при покупке и продаже)")
    print(f"  Депозиты: {optimizer.commission_deposit*100:.0f}% (при покупке и продаже)")
    print()
    
    print("Решение задачи...")
    print()
    
    try:
        result = optimizer.solve_with_trace()

        print(f"Максимальный ожидаемый доход (критерий Байеса): {result['optimal_expected_value']:.2f} д.е.")
        print()
        
        print("Оптимальная стратегия управления:")
        
        for i, (action, state) in enumerate(result['optimal_path']):
            print(f"\nЭтап {i + 1}:")
            print(f"  Действие:")
            if action.delta_cb1 != 0:
                print(f"    ЦБ1: {'+' if action.delta_cb1 > 0 else ''}{action.delta_cb1} шагов "
                      f"({action.delta_cb1 * optimizer.step_size_cb1:+.2f} д.е.)")
            if action.delta_cb2 != 0:
                print(f"    ЦБ2: {'+' if action.delta_cb2 > 0 else ''}{action.delta_cb2} шагов "
                      f"({action.delta_cb2 * optimizer.step_size_cb2:+.2f} д.е.)")
            if action.delta_deposit != 0:
                print(f"    Депозиты: {'+' if action.delta_deposit > 0 else ''}{action.delta_deposit} шагов "
                      f"({action.delta_deposit * optimizer.step_size_deposit:+.2f} д.е.)")
            if action.delta_cb1 == 0 and action.delta_cb2 == 0 and action.delta_deposit == 0:
                print("    Без изменений")
            
            print(f"  Состояние после действия:")
            print(f"    ЦБ1: {state.cb1:.2f} д.е.")
            print(f"    ЦБ2: {state.cb2:.2f} д.е.")
            print(f"    Депозиты: {state.deposit:.2f} д.е.")
            print(f"    Свободные средства: {state.free_cash:.2f} д.е.")
            print(f"    Общая стоимость: {state.total_value():.2f} д.е.")
        
        print(f"Итоговый ожидаемый доход: {result['optimal_expected_value']:.2f} д.е.")
        print(f"Прирост к начальной стоимости: {result['optimal_expected_value'] - initial.total_value():.2f} д.е.")
        print(f"Процент прироста: {(result['optimal_expected_value'] / initial.total_value() - 1) * 100:.2f}%")
        
    except Exception as e:
        print(f"Ошибка при решении задачи: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
