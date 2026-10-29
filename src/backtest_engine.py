import csv
import os
import pandas as pd
import numpy as np
import logging


class PortfolioOB:
    def __init__(self, board_panel: pd.DataFrame, volatility_panel: pd.DataFrame, volume_panel: pd.DataFrame,
                 delist_history: pd.DataFrame, strategies: list, commission: float = 0.0005):
        self.board_panel = board_panel.sort_index()
        self.strategies = strategies
        self.commission = commission
        self.daily_inflation = (1.0 + 0.075) ** (1.0 / 252.0) - 1.0
        self.asset_tickers = [col for col in self.board_panel.columns if not col.endswith('_div')]
        self.volatility_panel = volatility_panel.sort_index()
        self.volume_panel = volume_panel.sort_index()
        self.delist_history = delist_history
        self.lqdt = board_panel['Денежный рынок(LQDT)']

    def simulation(self, initial_capital: float, start_date: str = "2015-01-01", end_date: str = "2028-01-01",
                   key: bool = True, assets: str = None) -> pd.DataFrame:

        if assets:
            self.asset_tickers = [col for idx, col in enumerate(self.asset_tickers) if assets[idx] == '1']

        if not key:
            all_time_weight = {strat.name: np.zeros(len(self.asset_tickers)) for strat in self.strategies}
            all_time_turnover = {strat.name: 0.0 for strat in self.strategies}

        ticker_to_idx = {ticker: idx for idx, ticker in enumerate(self.asset_tickers)}
        sim_dates = self.board_panel[self.asset_tickers].loc[start_date:end_date].index
        T = len(sim_dates)
        last_t_in_global = self.lqdt.index.get_loc(sim_dates[-1])

        cols = []
        active_scenarios = ['Lump', 'Lump_tf'] if key else ['Lump', 'Lump_tf', 'DCA', 'FIRE']
        for strat in self.strategies:
            sn = strat.name
            for acsc in active_scenarios:
                cols.append(f'{sn}_{acsc}')
        if not key:
            cols.extend(['Inflation_Benchmark_Lump', 'Inflation_Benchmark_DCA', 'Inflation_Benchmark_FIRE'])

        res = pd.DataFrame(0.0, index=sim_dates, columns=cols)

        state = {
            strat.name: {
                scen: {
                    'capital': 50_000.0 if scen == 'DCA' else 6_000_000 if scen == 'FIRE' else initial_capital,
                    'prev_year_cap': 50_000.0 if scen == 'DCA' else 6_000_000 if scen == 'FIRE' else initial_capital,
                    'payout_queue': [],
                    'default' : False
                } for scen in active_scenarios
            } for strat in self.strategies
        }
        fix_сash_flow = {'Lump': 0.0, 'Lump_tf': 0.0, 'DCA': 50_000.0, 'FIRE': -60_000.0}
        benchmarks = {'Lump': initial_capital, 'DCA': 50_000.0, 'FIRE': 6_000_000.0}

        prev_weights = {
            strat.name: {
                scen: np.zeros(len(self.asset_tickers)) for scen in active_scenarios
            } for strat in self.strategies
        }

        for t_idx, current_date in enumerate(sim_dates):
            logging.info(f"Завершен рассчет на дату {current_date}")

            historical_slice = self.board_panel[self.asset_tickers].loc[:current_date]
            two_days = historical_slice.tail(2)
            current_day_returns = np.nan_to_num(historical_slice.iloc[-1].to_numpy(), nan=0.0)
            historical_slice = historical_slice.iloc[:-1]

            active_mask = ~two_days.isna().any().to_numpy()
            live_cols = [col for i, col in enumerate(self.asset_tickers) if active_mask[i]]

            cleaned_slice = historical_slice[live_cols]
            vol_slice = self.volatility_panel[live_cols].loc[:current_date].iloc[:-1]
            div_slice = self.board_panel[[f"{col}_div" for col in live_cols]].loc[:current_date].fillna(0.0)
            last_day_div = div_slice.iloc[-1]
            div_slice = div_slice.iloc[:-1]
            total_return_slice = cleaned_slice.to_numpy() + div_slice.to_numpy()
            historical_returns_adjusted = pd.DataFrame(total_return_slice, index=cleaned_slice.index, columns=live_cols)
            volume_slice = self.volume_panel[live_cols].loc[:current_date].iloc[:-1]

            bench_key = True
            if t_idx >= 2:
                is_new_month = sim_dates[t_idx - 1].month != sim_dates[t_idx - 2].month
                is_new_year = sim_dates[t_idx - 1].year != sim_dates[t_idx - 2].year
            else: is_new_month, is_new_year = False, False

            for strat in self.strategies:
                name = strat.name

                for scen in active_scenarios:
                    cash_flow = fix_сash_flow[scen] if is_new_month else 0.0
                    if bench_key and scen != 'Lump_tf':
                        benchmarks[scen] = benchmarks[scen] * (1.0 + self.daily_inflation) + cash_flow
                        if benchmarks[scen] <= 0: benchmarks[scen] = 0.0
                        res.loc[current_date, f'Inflation_Benchmark_{scen}'] = benchmarks[scen]

                    if state[name][scen]['default']: continue

                    prev_capital = state[name][scen]['capital']
                    prev_asset_values = prev_capital * prev_weights[name][scen]
                    tax = 0.0 if scen == 'Lump_tf' else 0.13

                    if is_new_year:
                        tax_base = prev_capital - state[name][scen]["prev_year_cap"]
                        if tax_base > 0:
                            tax_amount = tax_base * tax
                            prev_capital -= tax_amount
                        state[name][scen]["prev_year_cap"] = prev_capital

                    payout = 0.0
                    ready_2_pay = [item for item in state[name][scen]["payout_queue"] if item[0] <= t_idx]
                    state[name][scen]["payout_queue"] = [item for item in state[name][scen]["payout_queue"] if item[0] > t_idx]
                    for item in ready_2_pay:
                        payout += item[1]
                    prev_capital += payout
                    state[name][scen]["prev_year_cap"] += payout

                    prev_capital += cash_flow
                    state[name][scen]["prev_year_cap"] += cash_flow

                    if prev_capital <= 0:
                        if scen != 'DCA':
                            res.loc[current_date:, f'{name}_{scen}'] = 0.0
                            state[name][scen]["default"] = True
                        else: res.loc[current_date, f'{name}_{scen}'] = 0.0
                        continue

                    if scen != 'Lump_tf':
                        live_prev_weights = (prev_weights[name][scen])[active_mask].copy()
                        if np.sum(live_prev_weights) > 0: live_prev_weights = live_prev_weights / np.sum(live_prev_weights)

                        live_new_weights = strat.optimize_weights(historical_returns=historical_returns_adjusted,
                                                                  prev_weights=live_prev_weights,
                                                                  volatility_history=vol_slice,
                                                                  volume_history=volume_slice,
                                                                  current_portfolio_value=state[name][scen]['capital'])
                        live_new_weights = np.nan_to_num(live_new_weights, nan=0.0)

                    current_weights = np.zeros(len(self.asset_tickers))
                    current_weights[active_mask] = live_new_weights

                    if np.allclose(current_weights, prev_weights[name][scen], atol=2e-11, rtol=0): state[name][scen]['capital'] = prev_capital
                    else:
                        sign_vector = np.sign(prev_capital * current_weights - prev_asset_values)

                        for _ in range(5):
                            numerator = prev_capital + self.commission * np.sum(sign_vector * prev_asset_values)
                            denominator = 1.0 + self.commission * np.sum(sign_vector * current_weights)
                            solution_capital = numerator / denominator

                            new_sign_vector = np.sign(solution_capital * current_weights - prev_asset_values)
                            if np.array_equal(sign_vector, new_sign_vector): break
                            sign_vector = new_sign_vector

                        state[name][scen]['capital'] = solution_capital

                    if (not key) and (scen == 'Lump'):
                        all_time_weight[name] += current_weights
                        if t_idx > 0: all_time_turnover[name] += np.sum(np.abs(solution_capital * current_weights - prev_asset_values))

                    dividend_accrued = state[name][scen]['capital'] * np.sum(current_weights[active_mask] * last_day_div)
                    if dividend_accrued > 0:
                        if t_idx + 15 < T: state[name][scen]["payout_queue"].append((t_idx + 15, dividend_accrued))
                        elif (end_idx := last_t_in_global + (t_idx + 16 - T)) <= len(self.lqdt): state[name][scen]["payout_queue"].append((T - 1, dividend_accrued / np.prod(1 + self.lqdt.iloc[last_t_in_global + 1: end_idx])))
                        else: state[name][scen]["payout_queue"].append((T - 1, dividend_accrued / (np.mean(1 + self.lqdt.iloc[last_t_in_global - 4: last_t_in_global + 1])) ** (t_idx + 16 - T)))

                    if current_date in self.delist_history.index:
                        delist_row = self.delist_history.loc[current_date]
                        if (comp := str(delist_row['Company'])) in self.asset_tickers:
                            if t_idx + int(delist_row['Pay_lag']) < T: state[name][scen]["payout_queue"].append((t_idx + int(delist_row['Pay_lag']), state[name][scen]['capital'] * current_weights[ticker_to_idx[comp]] * float(delist_row['Amount'])))
                            elif (end_idx := last_t_in_global + (t_idx + 1 + int(delist_row['Pay_lag']) - T)) <= len(self.lqdt): state[name][scen]["payout_queue"].append((T - 1, state[name][scen]['capital'] * current_weights[ticker_to_idx[comp]] * float(delist_row['Amount']) / np.prod(1 + self.lqdt.iloc[last_t_in_global + 1: end_idx])))
                            else: state[name][scen]["payout_queue"].append((T - 1, state[name][scen]['capital'] * current_weights[ticker_to_idx[comp]] * float(delist_row['Amount']) / (np.mean(1 + self.lqdt.iloc[last_t_in_global - 4: last_t_in_global + 1])) ** (t_idx + 1 + int(delist_row['Pay_lag']) - T)))
                            if float(delist_row['Amount']) > 0: state[name][scen]["prev_year_cap"] -= state[name][scen]['capital'] * current_weights[ticker_to_idx[comp]]

                    capital_growth = np.sum(current_weights * current_day_returns)
                    state[name][scen]['capital'] *= (1.0 + capital_growth)
                    res.loc[current_date, f'{name}_{scen}'] = state[name][scen]['capital']

                    prev_weights[name][scen] = current_weights * (1.0 + current_day_returns)
                    prev_weights[name][scen] = prev_weights[name][scen] / np.sum(prev_weights[name][scen])
                bench_key = False

            for scen in active_scenarios: fix_сash_flow[scen] *= (1.0 + self.daily_inflation)

        if not key:
            new = not os.path.isfile('data/results/strat_csv/bonusinfo.csv')


            with open('data/results/strat_csv/bonusinfo.csv', mode='a', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                if new: w.writerow(['name', 'turnover'] + self.asset_tickers)
                for strat in self.strategies: w.writerow([strat.name, all_time_turnover[strat.name]] + list(all_time_weight[strat.name]/T))

        return res