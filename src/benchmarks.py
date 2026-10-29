import numpy as np
import pandas as pd
from empty_box import BasePortfolioStrategy
import cvxpy as cp
from sklearn.linear_model import Ridge
from sklearn.covariance import ledoit_wolf
from lightgbm import LGBMRegressor
import scipy.stats as stats

commission = 0.0005

# Дети

class UniformStrategy(BasePortfolioStrategy):
    def __init__(self):
        super().__init__(name="Uniform 1/N")

    def optimize_weights(self, historical_returns: pd.DataFrame, prev_weights: np.ndarray, volatility_history: pd.DataFrame,
                         volume_history: pd.DataFrame, current_portfolio_value: float
                         ) -> np.ndarray:
        N = historical_returns.shape[1]
        weights = np.ones(N) / N
        return weights


class RandomMonkeyStrategy(BasePortfolioStrategy):
    def __init__(self):
        super().__init__(name="Random Monkey")

    def optimize_weights(self, historical_returns: pd.DataFrame, prev_weights: np.ndarray, volatility_history: pd.DataFrame,
                         volume_history: pd.DataFrame, current_portfolio_value: float) -> np.ndarray:
        if (historical_returns.index[-1].month != historical_returns.index[-2].month) or (np.sum(prev_weights) == 0):
            N = historical_returns.shape[1]
            random_vectors = np.random.rand(N)
            weights = random_vectors / np.sum(random_vectors)
        else: weights = prev_weights

        return weights


class StochasticMomentumStrategy(BasePortfolioStrategy):
    def __init__(self, lookback_days: int = 63, num_pick_assets: int = 10):
        super().__init__(name="Stochastic Momentum")
        self.lookback = lookback_days
        self.num_picks = num_pick_assets

    def optimize_weights(self, historical_returns: pd.DataFrame, prev_weights: np.ndarray, volatility_history: pd.DataFrame,
                         volume_history: pd.DataFrame, current_portfolio_value: float) -> np.ndarray:
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

# Юноши

class MarkowitzStrategy(BasePortfolioStrategy):
    def __init__(self, target_daily_return: float = 0.0004):
        super().__init__(name="Markowitz")
        self.target_return = target_daily_return
        self.commission = commission

    def optimize_weights(self, historical_returns: pd.DataFrame, prev_weights: np.ndarray, volatility_history: pd.DataFrame,
                         volume_history: pd.DataFrame, current_portfolio_value: float) -> np.ndarray:
        N = len(historical_returns.columns)
        columns = list(historical_returns.columns)
        df_live = historical_returns.tail(252)

        returns_np = df_live.to_numpy()
        volume_np = volume_history.to_numpy()
        mean_22 = np.mean(volume_np[-22:], axis=0)
        mean_5 = np.mean(volume_np[-5:], axis=0)
        max_liquidity_weights = (0.015 * (0.8 * np.minimum(mean_5, mean_22) + 0.2 * np.maximum(mean_5, mean_22))) / current_portfolio_value
        freeze = (prev_weights - (volume_np[-1] * 0.1 / current_portfolio_value) > max_liquidity_weights)
        max_liquidity_weights[freeze] = prev_weights[freeze] - (volume_np[-1][freeze] * 0.1 / current_portfolio_value)

        mean_returns = np.nanmean(returns_np, axis=0)
        max_day_delta = 0.1

        returns_clean = np.nan_to_num(returns_np, nan=0.0)
        sigma_live, _ = ledoit_wolf(returns_clean)

        w = cp.Variable(N)
        portfolio_risk = cp.quad_form(w, sigma_live)

        turnover_penalty = self.commission * cp.sum(cp.abs(w - prev_weights))

        objective = cp.Minimize(portfolio_risk + turnover_penalty)
        constraints = [
            cp.sum(w) == 1.0,
            w >= 0.0,
            w <= max_liquidity_weights
        ]

        currency_indices = [i for i, name in enumerate(columns) if name in ['Юань', 'Доллар', 'Евро']]
        metal_indices = [i for i, name in enumerate(columns) if name in ['Золото', 'Серебро']]

        if currency_indices: constraints.append(cp.sum(w[currency_indices]) <= 0.20)
        if metal_indices: constraints.append(cp.sum(w[metal_indices]) <= 0.15)
        constraints.append(w[[i for i, name in enumerate(columns) if name == 'Денежный рынок(LQDT)'][0]] >= 0.05)

        if np.max(mean_returns) >= self.target_return: constraints.append(mean_returns @ w >= self.target_return)
        elif (median_returns := np.median(mean_returns)) > 0: constraints.append(mean_returns @ w >= median_returns)
        else: max_day_delta = 0.3

        if np.nansum(prev_weights) > 0:
            max_trade_delta_weights = max_day_delta * volume_np[-1] / current_portfolio_value
            constraints.append(cp.abs(w - prev_weights) <= max_trade_delta_weights)

        prob = cp.Problem(objective, constraints)
        prob.solve()

        return w.value


class BaseCVaRStrategy(BasePortfolioStrategy):
    def __init__(self):
        super().__init__(name="Base CVaR")
        self.commission = commission

    def optimize_weights(self, historical_returns: pd.DataFrame, prev_weights: np.ndarray, volatility_history: pd.DataFrame,
                         volume_history: pd.DataFrame, current_portfolio_value: float) -> np.ndarray:
        columns = list(historical_returns.columns)
        N = len(columns)
        safe_names = ['Денежный рынок(LQDT)']

        df_window = historical_returns.tail(252)
        returns_np = df_window.to_numpy()
        clean_returns_np = np.nan_to_num(returns_np, nan=0.0)
        T_window = len(clean_returns_np)

        volume_np = volume_history.to_numpy()
        mean_22 = np.mean(volume_np[-22:], axis=0)
        mean_5 = np.mean(volume_np[-5:], axis=0)
        max_liquidity_weights = (0.015 * (0.8 * np.minimum(mean_5, mean_22) + 0.2 * np.maximum(mean_5, mean_22))) / current_portfolio_value
        max_liquidity_weights = np.minimum(max_liquidity_weights, [1.0 if (col in safe_names) else 0.1 for col in columns])
        freeze = (prev_weights - (volume_np[-1] * 0.1 / current_portfolio_value) > max_liquidity_weights)
        max_liquidity_weights[freeze] = prev_weights[freeze] - (volume_np[-1][freeze] * 0.1 / current_portfolio_value)

        max_day_delta = 0.1
        beta = 0.95
        w = cp.Variable(N)
        alpha = cp.Variable()
        u = cp.Variable(T_window)

        losses = -clean_returns_np @ w
        cvar_loss = alpha + (1.0 / (1.0 - beta)) * cp.mean(u)

        turnover_penalty = self.commission * cp.sum(cp.abs(w - prev_weights))

        lqdt = [i for i, col in enumerate(columns) if col in safe_names]
        objective = cp.Minimize(cvar_loss + turnover_penalty + cp.abs((w[lqdt] - 0.4)))

        constraints = [
            cp.sum(w) == 1.0,
            w >= 0.0,
            u >= 0.0,
            u >= losses - alpha,
            w <= max_liquidity_weights
        ]

        currency_indices = [i for i, name in enumerate(columns) if name in ['Юань', 'Доллар', 'Евро']]
        metal_indices = [i for i, name in enumerate(columns) if name in ['Золото', 'Серебро']]

        if currency_indices: constraints.append(cp.sum(w[currency_indices]) <= 0.20)
        if metal_indices: constraints.append(cp.sum(w[metal_indices]) <= 0.15)
        constraints.append(w[[i for i, name in enumerate(columns) if name == 'Денежный рынок(LQDT)'][0]] >= 0.05)

        if np.median(np.nanmean(returns_np, axis=0)) <= 0: max_day_delta = 0.3

        if np.nansum(prev_weights) > 0:
            max_trade_delta_weights = max_day_delta * volume_np[-1] / current_portfolio_value
            constraints.append(cp.abs(w - prev_weights) <= max_trade_delta_weights)

        prob = cp.Problem(objective, constraints)
        prob.solve(warm_start=True)

        return w.value


class ML_RankEnsemble_Frankenstein(BasePortfolioStrategy):
    def __init__(self, name: str = "RankEnsemble_Frankenstein"):
        super().__init__(name=name)
        self.commission = commission
        self.max_horizon = 500 + 5 + 1
        self.eval_horizon = 63
        self.history_ridge_preds = []
        self.history_lgbm_preds = []
        self.history_real_returns = []

    def _get_ridge_predictions(self, train_returns: np.ndarray, train_vol: np.ndarray, train_volume: np.ndarray) -> list[np.ndarray]:
        lag_1 = train_returns[4:-1]
        lag_2 = train_returns[3:-2]
        lag_5 = train_returns[:-5]
        vol = train_vol[4:-1]
        vol_lag1 = train_vol[3:-2]
        log_volume = np.log1p(train_volume[4:-1])

        target_raw = train_returns[5:]
        daily_mean = np.mean(target_raw, axis=1, keepdims=True)
        daily_std = np.std(target_raw, axis=1, keepdims=True) + 1e-8
        target = (target_raw - daily_mean) / daily_std

        num_days, _ = lag_1.shape
        num_features = 6
        preds = []

        beta_history = np.zeros((num_days, num_features))
        intercept_history = np.zeros(num_days)

        for t in range(num_days):
            X_t = np.column_stack([lag_1[t], lag_2[t], lag_5[t], vol[t], vol_lag1[t], log_volume[t]])
            y_t = target[t]

            mean_X = np.mean(X_t, axis=0)
            std_X = np.std(X_t, axis=0)

            X_t_scaled = (X_t - mean_X) / (std_X + 1e-6)

            model = Ridge(alpha=10.0)
            model.fit(X_t_scaled, y_t)

            beta_history[t] = model.coef_ / (std_X + 1e-6)
            intercept_history[t] = model.intercept_ - np.sum(model.coef_ * mean_X / (std_X + 1e-6))

        weights = np.power(0.95, np.arange(num_days - 1)[::-1])
        weights /= np.sum(weights)

        for i in range(2):
            smooth_beta = np.sum(beta_history[i:(-1 if i == 0 else None)] * weights[:, np.newaxis], axis=0)
            smooth_intercept = np.sum(intercept_history[i:(-1 if i == 0 else None)] * weights)

            X_predict = np.column_stack([
                train_returns[i - 2], train_returns[i - 3], train_returns[i - 6],
                train_vol[i - 2], train_vol[i - 3], np.log1p(train_volume[i - 2])
            ])
            preds.append(X_predict @ smooth_beta + smooth_intercept)

        return preds

    def _get_lgbm_predictions(self, train_returns: np.ndarray, train_vol: np.ndarray, train_volume: np.ndarray) -> np.ndarray:
        lag_1 = train_returns[4:-1]
        lag_2 = train_returns[3:-2]
        lag_5 = train_returns[:-5]
        vol = train_vol[4:-1]
        vol_lag1 = train_vol[3:-2]
        log_volume = np.log1p(train_volume[4:-1])

        daily_market_return = np.mean(train_returns, axis=1)
        return_rolling_5 = np.convolve(daily_market_return, np.ones(5) / 5.0, mode='valid')
        market_macro_panel = np.repeat(return_rolling_5[:-1], lag_1.shape[1])

        X_train = np.column_stack([
            lag_1.ravel(), lag_2.ravel(), lag_5.ravel(), vol.ravel(),
            vol_lag1.ravel(), log_volume.ravel(), market_macro_panel
        ])

        target_raw = train_returns[5:]
        daily_mean = np.mean(target_raw, axis=1, keepdims=True)
        daily_std = np.std(target_raw, axis=1, keepdims=True)
        target_z = (target_raw - daily_mean) / (daily_std + 1e-6)
        y_train = target_z.ravel()

        X_predict = np.column_stack([
            train_returns[-1], train_returns[-2], train_returns[-5], train_vol[-1], train_vol[-2],
            np.log1p(train_volume[-1]), np.full(train_returns.shape[1], return_rolling_5[-1])
        ])

        model = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.02,
            max_depth=4,
            min_child_samples=50,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        num_days = lag_1.shape[0]
        day_weights = np.power(0.995, np.arange(num_days)[::-1])
        sample_weights = np.repeat(day_weights, train_returns.shape[1])

        model.fit(X_train, y_train, sample_weight=sample_weights)

        preds = model.predict(X_predict)
        return np.nan_to_num(preds, nan=0.0)

    def _compute_ranks(self, predictions: np.ndarray) -> np.ndarray:
        ranks = np.argsort(np.argsort(predictions))
        return ranks / np.max(ranks)

    def optimize_weights(self, historical_returns: pd.DataFrame, prev_weights: np.ndarray, volatility_history: pd.DataFrame,
                         volume_history: pd.DataFrame, current_portfolio_value: float) -> np.ndarray:
        columns = list(historical_returns.columns)
        N = len(columns)
        safe_names = ['Денежный рынок(LQDT)']
        ema_window = 5

        df_window = historical_returns.tail(self.max_horizon)
        returns_np = df_window.to_numpy()
        clean_returns_np = np.nan_to_num(returns_np, nan=0.0)

        df_vol_window = volatility_history.tail(self.max_horizon)
        vol_matrix_np = np.nan_to_num(df_vol_window.to_numpy(), nan=0.0005)
        daily_market_vol = np.mean(vol_matrix_np, axis=1)

        volume_np = volume_history.tail(self.max_horizon).to_numpy()
        mean_22 = np.mean(volume_np[-22:], axis=0)
        mean_5 = np.mean(volume_np[-5:], axis=0)
        max_liquidity_weights = (0.015 * (0.8 * np.minimum(mean_5, mean_22) + 0.2 * np.maximum(mean_5, mean_22))) / current_portfolio_value
        max_liquidity_weights = np.minimum(max_liquidity_weights, [1.0 if (col in safe_names) else 0.1 for col in columns])
        freeze = (prev_weights - (volume_np[-1] * 0.1 / current_portfolio_value) > max_liquidity_weights)
        max_liquidity_weights[freeze] = prev_weights[freeze] - (volume_np[-1][freeze] * 0.1 / current_portfolio_value)

        w_ridge, w_lgbm = 0.5, 0.5

        preds_ridge_last, preds_ridge = self._get_ridge_predictions(clean_returns_np, vol_matrix_np, volume_np)
        preds_lgbm_last = self._get_lgbm_predictions(clean_returns_np[:-1], vol_matrix_np[:-1], volume_np[:-1])

        self.history_real_returns.append(returns_np[-1])
        self.history_ridge_preds.append(preds_ridge_last)
        self.history_lgbm_preds.append(preds_lgbm_last)

        if len(self.history_real_returns) > self.eval_horizon:
            self.history_real_returns.pop(0)
            self.history_ridge_preds.pop(0)
            self.history_lgbm_preds.pop(0)

        if len(self.history_real_returns) >= 15:
            ic_ridge_list = []
            ic_lgbm_list = []

            for t in range(len(self.history_real_returns)):
                real = self.history_real_returns[t]

                ic_ridge = np.corrcoef(self.history_ridge_preds[t], real)[0, 1]
                ic_lgbm = np.corrcoef(self.history_lgbm_preds[t], real)[0, 1]

                ic_ridge_list.append(ic_ridge)
                ic_lgbm_list.append(ic_lgbm)

            ic_ridge_list = np.nan_to_num(np.array(ic_ridge_list), nan=0.0)
            ic_lgbm_list = np.nan_to_num(np.array(ic_lgbm_list), nan=0.0)

            mean_ic_ridge = np.mean(ic_ridge_list)
            mean_ic_lgbm = np.mean(ic_lgbm_list)

            exp_ridge = np.exp(mean_ic_ridge * 20.0)
            exp_lgbm = np.exp(mean_ic_lgbm * 20.0)

            w_ridge = exp_ridge / (exp_ridge + exp_lgbm)
            w_lgbm = exp_lgbm / (exp_ridge + exp_lgbm)

        preds_lgbm = self._get_lgbm_predictions(clean_returns_np[1:], vol_matrix_np[1:], volume_np[1:])

        ranks_ridge = self._compute_ranks(preds_ridge)
        ranks_lgbm = self._compute_ranks(preds_lgbm)

        predict_rank = w_ridge * ranks_ridge + w_lgbm * ranks_lgbm

        predict_rank_clipped = np.clip(predict_rank, 1e-3, 1.0 - 1e-3)
        z_scores_gaussian = stats.norm.ppf(predict_rank_clipped)

        ema_weights = np.exp(np.linspace(-1, 0, ema_window))
        ema_weights /= np.sum(ema_weights)
        fast_market_vol = np.sum(daily_market_vol[-ema_window:] * ema_weights)
        predict_ret = 0.035 * fast_market_vol * z_scores_gaussian

        max_day_delta = 0.1
        w = cp.Variable(N)
        expected_return = predict_ret @ w
        turnover_penalty = self.commission * cp.sum(cp.abs(w - prev_weights))

        objective = cp.Maximize(expected_return - turnover_penalty)

        constraints = [
            cp.sum(w) == 1.0,
            w >= 0.0,
            w <= max_liquidity_weights
        ]

        currency_indices = [i for i, name in enumerate(columns) if name in ['Юань', 'Доллар', 'Евро']]
        metal_indices = [i for i, name in enumerate(columns) if name in ['Золото', 'Серебро']]

        if currency_indices: constraints.append(cp.sum(w[currency_indices]) <= 0.20)
        if metal_indices: constraints.append(cp.sum(w[metal_indices]) <= 0.15)
        constraints.append(w[[i for i, name in enumerate(columns) if name == 'Денежный рынок(LQDT)'][0]] >= 0.05)

        if np.nansum(prev_weights) > 0:
            max_trade_delta_weights = max_day_delta * volume_np[-1] / current_portfolio_value
            constraints.append(cp.abs(w - prev_weights) <= max_trade_delta_weights)

        prob = cp.Problem(objective, constraints)
        prob.solve(warm_start=True)
        return w.value


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
