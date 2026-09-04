import numpy as np
import pandas as pd
from empty_box import BasePortfolioStrategy
import cvxpy as cp
from sklearn.svm import LinearSVR
from sklearn.linear_model import Ridge
from sklearn.covariance import ledoit_wolf

commission = 0.0005


class UniformStrategy(BasePortfolioStrategy):
    """Бенчмарк 1: Равновзвешенный инвестор.
    Делит капитал поровну между всеми доступными на момент ребалансировки активами."""
    def __init__(self):
        super().__init__(name="Uniform 1/N")

    def optimize_weights(self, historical_returns: pd.DataFrame, prev_weights: np.ndarray, volatility_history: pd.DataFrame = None) -> np.ndarray:
        N = historical_returns.shape[1]
        weights = np.ones(N) / N
        return weights


class RandomMonkeyStrategy(BasePortfolioStrategy):
    """Бенчмарк 2: Случайный инвестор.
    Генерирует случайные веса на каждой ребалансировке. Ребалансировки реже обычного."""
    def __init__(self):
        super().__init__(name="Random Monkey")
        self._days_passed = 22

    def optimize_weights(self, historical_returns: pd.DataFrame, prev_weights: np.ndarray, volatility_history: pd.DataFrame = None) -> np.ndarray:
        if self._days_passed // 22:
            N = historical_returns.shape[1]
            random_vectors = np.random.rand(N)
            weights = random_vectors / np.sum(random_vectors)
            self._days_passed = 0
        else:
            self._days_passed += 1
            weights = prev_weights

        return weights


class StochasticMomentumStrategy(BasePortfolioStrategy):
    """Бенчмарк 3: Детерминированный Ротатор Трендов (Линейно-Взвешенный).
    Находит до Топ-10 активов с максимальным положительным ростом за последние 3 месяца
    и распределяет капитал между ними пропорционально силе их тренда."""
    def __init__(self, lookback_days: int = 63, num_pick_assets: int = 10):
        super().__init__(name="Stochastic Momentum")
        self.lookback = lookback_days
        self.num_picks = num_pick_assets

    def optimize_weights(self, historical_returns: pd.DataFrame, prev_weights: np.ndarray, volatility_history: pd.DataFrame = None) -> np.ndarray:
        N = historical_returns.shape[1]
        weights = np.zeros(N)

        recent_history = historical_returns.tail(self.lookback).to_numpy()
        cum_returns = np.nanprod(1.0 + recent_history, axis=0) - 1.0

        positive_indices = np.where(cum_returns > 0.0)[0]

        if len(positive_indices) == 0:
            return np.ones(N) / N

        positive_returns = cum_returns[positive_indices]
        top_sub_indices = np.argsort(positive_returns)[-self.num_picks:]
        chosen_indices = positive_indices[top_sub_indices]

        chosen_returns = cum_returns[chosen_indices]
        sum_chosen_returns = np.sum(chosen_returns)
        weights[chosen_indices] = chosen_returns / sum_chosen_returns
        return weights


class PersonalProfileStrategy(BasePortfolioStrategy):
    """Бенчмарк 4: Мой Инвестиционный Профиль.
    Распределяет капитал по жесткой иерархии классов активов: Возраст% в ОФЗ/Корпораты и возможно в ВДО,
    5% в Металлы по среднему распределению, Валютный коридор (переключение на Юань в 2022) и остаток в Акции."""
    def __init__(self, age_weight: float = 0.2, currency_weight: float = 0.10, lookback_days: int = 63):
        super().__init__(name="Personal Profile")
        self.age_w = age_weight
        self.fx_w = currency_weight
        self.metals_w = 0.05
        self.lookback = lookback_days

    def optimize_weights(self, historical_returns: pd.DataFrame, prev_weights: np.ndarray, volatility_history: pd.DataFrame = None) -> np.ndarray:
        columns = list(historical_returns.columns)
        N = len(columns)
        weights = np.zeros(N)

        if len(historical_returns) < self.lookback:
            return np.ones(N) / N

        current_date = historical_returns.index[-1]
        recent_history = historical_returns.tail(self.lookback)

        equity_cols = [c for c in columns if c not in ['ОФЗ, фикс 1+', 'ОФЗ, фикс 5-10', 'ВДО, фикс', 'Денежный рынок(LQDT)',
                                                       'Доллар', 'Евро', 'Юань', 'Недвижимость', 'Золото', 'Серебро']]
        market_trend = 0.0
        if equity_cols:
            market_trend = np.nanmean(np.nanprod(1.0 + recent_history[equity_cols].to_numpy(), axis=0) - 1.0)

        if market_trend >= 0.10:
            current_cash_w = 0.10
        elif market_trend <= -0.10:
            current_cash_w = 0.0
        else:
            current_cash_w = 0.05 + (market_trend / 0.10) * 0.05

        if 'Денежный рынок(LQDT)' in columns:
            weights[columns.index('Денежный рынок(LQDT)')] = current_cash_w

            lqdt_ret = recent_history['Денежный рынок(LQDT)'].to_numpy()
            half = len(lqdt_ret) // 2
            rate_momentum = np.nansum(lqdt_ret[half:]) - np.nansum(lqdt_ret[:half])

            if rate_momentum > 0.0001:
                w_short, w_long, w_high_yield = 0.8, 0.1, 0.1
            elif rate_momentum < -0.0001:
                w_short, w_long, w_high_yield = 0.2, 0.7, 0.1
            elif lqdt_ret[-1] < (1.06) ** (1 / 250) - 1:
                w_short, w_long, w_high_yield = 0.3, 0.2, 0.5
            else:
                w_short, w_long, w_high_yield = 0.4, 0.6, 0.0

            if 'ОФЗ, фикс 1+' in columns:
                weights[columns.index('ОФЗ, фикс 1+')] = w_short * self.age_w
            if 'ОФЗ, фикс 5-10' in columns:
                weights[columns.index('ОФЗ, фикс 5-10')] = w_long * self.age_w
            if 'ВДО, фикс' in columns:
                weights[columns.index('ВДО, фикс')] = w_high_yield * self.age_w

        active_fx = 'Доллар' if current_date < pd.Timestamp('2022-06-01') else 'Юань'

        if active_fx in columns:
            global_fx_series = np.cumprod(1.0 + historical_returns[active_fx].to_numpy())
            fx_curr = global_fx_series[-1]
            fx_mean = np.mean(global_fx_series[-self.lookback:])

            if fx_curr <= fx_mean:
                fx_target_w = self.fx_w
            else:
                deviation = (fx_curr - fx_mean) / fx_mean
                fx_target_w = max(0.0, self.fx_w * (1.0 - deviation / 0.10))

            weights[columns.index(active_fx)] = fx_target_w

        if 'Золото' in columns and 'Серебро' in columns:
            idx_g = columns.index('Золото')
            idx_s = columns.index('Серебро')

            g_prices = np.cumprod(1.0 + recent_history['Золото'].to_numpy())
            s_prices = np.cumprod(1.0 + recent_history['Серебро'].to_numpy())

            met_curr = g_prices[-1] / s_prices[-1]

            l, m, h = 0.02, 0.03, 0.045
            if met_curr < 1:
                weights[idx_g] = min(h, m + (1.0 - met_curr) * (h - m) / 0.3)
            else:
                weights[idx_g] = max(l, m - (met_curr - 1) * (m - l) / 0.3)

            weights[idx_s] = self.metals_w - weights[idx_g]

        allocated_cash = np.sum(weights)
        remaining_cash = 1.0 - allocated_cash

        if remaining_cash > 0 and equity_cols:
            equity_indices = [columns.index(c) for c in equity_cols]
            weights[equity_indices] = remaining_cash / len(equity_indices)

        return weights / np.sum(weights)


class MarkowitzStrategy(BasePortfolioStrategy):
    """Бенчмарк 5: Классическая Mean-Variance оптимизация Марковица в CVXPY.
    Минимизирует риск портфеля с учетом транзакционных издержек (L1-штраф), робастная оценка ковариации: используется сжатие Ледойта-Вулфа.
    Ограничения концентрации: жесткие лимиты на веса (до 10% в одну акцию, до 40% в кэш LQDT). Динамический Fallback: если целевая доходность недостижима, модель автоматически
    переключается на ограничение по медианной доходности пула активов."""
    def __init__(self, target_daily_return: float = 0.0004):
        super().__init__(name="Markowitz")
        self.target_return = target_daily_return
        self.commission = commission

    def optimize_weights(self, historical_returns: pd.DataFrame, prev_weights: np.ndarray, volatility_history: pd.DataFrame = None) -> np.ndarray:
        N = len(historical_returns.columns)

        df_live = historical_returns.tail(252)
        returns_np = df_live.to_numpy()

        mean_returns = np.nanmean(returns_np, axis=0)

        returns_clean = np.nan_to_num(returns_np, nan=0.0)
        sigma_live, _ = ledoit_wolf(returns_clean)

        w = cp.Variable(N)
        portfolio_risk = cp.quad_form(w, sigma_live)

        turnover_penalty = self.commission * cp.sum(cp.abs(w - prev_weights))

        objective = cp.Minimize(portfolio_risk + turnover_penalty)
        constraints = [
            cp.sum(w) == 1.0,
            w >= 0.0
        ]

        if np.max(mean_returns) >= self.target_return:
            constraints.append(mean_returns @ w >= self.target_return)
        elif (median_returns := np.median(mean_returns)) > 0:
            constraints.append(mean_returns @ w >= median_returns)

        safe_names = ['Денежный рынок(LQDT)']
        columns = list(historical_returns.columns)

        for i, col in enumerate(columns):
            if col not in safe_names: constraints.append(w[i] <= 0.1)
            else: constraints.append(w[i] <= 0.4)

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()

            if w.value is None or prob.status not in ['optimal', 'optimal_inaccurate']:
                raise ValueError("Несовместимые ограничения")
            return w.value
        except Exception:
            if np.nansum(prev_weights) > 0: return prev_weights
            else: return np.ones(N) / N


class BaseCVaRStrategy(BasePortfolioStrategy):
    """Классическая не-робастная оптимизация хвостового риска CVaR (Рокфеллер - Урьясев).
    Минимизирует ожидаемые потери в 5% худших сценариев с учетом транзакционных издержек через штраф на оборот относительно предыдущих весов.
    За счет жестких лимитов весов (40% LQDT / 10% акции) модель вынуждена распределять остаток капитала в акции с минимальным историческим хвостовым риском, 
    вместо полной капитуляции в фонд денежного рынка."""
    def __init__(self):
        super().__init__(name="Base CVaR")
        self.commission = commission

    def optimize_weights(self, historical_returns: pd.DataFrame, prev_weights: np.ndarray, volatility_history: pd.DataFrame = None) -> np.ndarray:
        columns = list(historical_returns.columns)
        N = len(columns)

        df_window = historical_returns.tail(252)
        returns_np = df_window.to_numpy()
        clean_returns_np = np.nan_to_num(returns_np, nan=0.0)
        T_window = len(clean_returns_np)

        beta = 0.95
        w = cp.Variable(N)
        alpha = cp.Variable()
        u = cp.Variable(T_window)

        losses = -clean_returns_np @ w
        cvar_loss = alpha + (1.0 / (1.0 - beta)) * cp.mean(u)

        turnover_penalty = self.commission * cp.sum(cp.abs(w - prev_weights))

        objective = cp.Minimize(cvar_loss + turnover_penalty)

        constraints = [
            cp.sum(w) == 1.0,
            w >= 0.0,
            u >= 0.0,
            u >= losses - alpha
        ]

        safe_names = ['Денежный рынок(LQDT)']
        columns = list(historical_returns.columns)

        for i, col in enumerate(columns):
            if col not in safe_names: constraints.append(w[i] <= 0.1)
            else: constraints.append(w[i] <= 0.4)

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()

            if w.value is None or prob.status not in ['optimal', 'optimal_inaccurate']:
                raise ValueError("Несовместимые ограничения")
            return w.value
        except Exception:
            if np.nansum(prev_weights) > 0:
                return prev_weights
            else:
                return np.ones(N) / N


class RobustParabolicCvarStrategy_new(BasePortfolioStrategy):
    def __init__(self, name: str = "Controlled-risk prototype"):
        super().__init__(name=name)
        self.commission = commission
        self.max_horizon = 1000

    def _generate_parabolic_kernel_weights(self, returns_np: np.ndarray, vol_matrix_np: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class RobustParabolicCvarStrategy_old(BasePortfolioStrategy):
    def __init__(self, name: str = "High-risk prototype"):
        super().__init__(name=name)
        self.commission = commission
        self.max_horizon = 1000

    def _generate_parabolic_kernel_weights(self, returns_np: np.ndarray, vol_matrix_np: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class RobustParabolicCvarStrategy_medium(BasePortfolioStrategy):
    def __init__(self, name: str = "0.1 prototype"):
        super().__init__(name=name)
        self.commission = commission
        self.max_horizon = 1000

    def _generate_parabolic_kernel_weights(self, returns_np: np.ndarray, vol_matrix_np: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
