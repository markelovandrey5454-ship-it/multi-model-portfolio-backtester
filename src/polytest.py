import pandas as pd
from datetime import datetime
import random
import numpy as np
import logging
from backtest_engine import PortfolioOrchestrator, PortfolioBacktester
import os
import csv
from benchmarks import (UniformStrategy, RandomMonkeyStrategy, MarkowitzStrategy, BaseCVaRStrategy,
                        RobustParabolicCvarStrategy_old, RobustParabolicCvarStrategy_new, RobustParabolicCvarStrategy_medium)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
BASE32_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUV"
n = 250

if __name__ == "__main__":
    panel_path = 'data/matrix/global_board_panel.csv'
    vol_path = 'data/matrix/global_volatility_panel.csv'
    del_path = 'data/matrix/global_delist_panel.csv'

    board_panel = pd.read_csv(panel_path, parse_dates=['Date']).set_index('Date')
    volatility_panel = pd.read_csv(vol_path, parse_dates=['Date']).set_index('Date')
    delist_history = pd.read_csv(del_path, parse_dates=['Date']).set_index('Date')
    print("=== ЗАПУСК СЛУЧАЙНОГО БЭКТЕСТА СИСТЕМЫ ===")
    for exper in range(1, n + 1):
        delt_date = random.randint(3,5)
        start_date = random.randint(int(datetime(2015, 1, 1).timestamp()), int(datetime(2026 - delt_date, 9, 1).timestamp()))
        start_date = pd.to_datetime(datetime.fromtimestamp(start_date).date())
        end_date = start_date + pd.DateOffset(years=delt_date)
        start = start_date.strftime("%Y-%m-%d")
        end = end_date.strftime("%Y-%m-%d")

        final_bit_mask = np.zeros(len(volatility_panel.columns), dtype=int)
        alive_assets_mask = np.isnan(volatility_panel.loc[start:end]).sum(axis=0) == 0
        alive_indices = np.where(alive_assets_mask)[0]

        lqdt_index = volatility_panel.columns.get_loc('Денежный рынок(LQDT)')
        final_bit_mask[lqdt_index] = 1

        gold_index = volatility_panel.columns.get_loc('Золото')
        silv_index = volatility_panel.columns.get_loc('Серебро')
        ofz1_index = volatility_panel.columns.get_loc('ОФЗ, фикс 1+')
        ofz5_index = volatility_panel.columns.get_loc('ОФЗ, фикс 5-10')

        excluded_indices = [lqdt_index, gold_index, silv_index, ofz1_index, ofz5_index]
        alive_indices = alive_indices[~np.isin(alive_indices, excluded_indices)]
        random_bits = np.random.randint(0, 2, size=len(alive_indices))
        final_bit_mask[alive_indices] = random_bits

        random_metl = np.random.randint(0, 2)
        final_bit_mask[gold_index] = random_metl
        final_bit_mask[silv_index] = 1 - random_metl

        random_ofz = np.random.randint(0, 2)
        final_bit_mask[ofz1_index] = random_ofz
        final_bit_mask[ofz5_index] = 1 - random_ofz

        assets_str = "".join(final_bit_mask.astype(str))
        encoded_assets = ''
        for i in range(0, len(final_bit_mask), 5):
            chunk = final_bit_mask[i:i + 5]
            val = 0
            for bit in chunk:
                val = (val << 1) | bit
            encoded_assets += BASE32_CHARS[val]

        strategies = [UniformStrategy(), RandomMonkeyStrategy(), MarkowitzStrategy(), BaseCVaRStrategy(), RobustParabolicCvarStrategy_old(), RobustParabolicCvarStrategy_new(), RobustParabolicCvarStrategy_medium()]

        print(f"\n[ШАГ 1/3] Запуск динамической симуляции ребалансировок моделей, эксперимент {exper}")
        orchestrator = PortfolioOrchestrator(board_panel, volatility_panel, strategies)
        all_weights = orchestrator.generate_weights_history(start_date=start, end_date=end, assets=assets_str)

        chosen_live_tickers = all_weights[strategies[-1].name].columns.tolist()
        chosen_board_cols = []
        for ticker in chosen_live_tickers:
            chosen_board_cols.append(ticker)
            div_col = f"{ticker}_div"
            if div_col in board_panel.columns:
                chosen_board_cols.append(div_col)

        print(f"\n[ШАГ 2/3] Расчет накопления капитала по макро-сценариям, эксперимент {exper}")
        backtester = PortfolioBacktester(board_panel[chosen_board_cols], delist_history, inflation_annual=0.075, commission=0.0005)

        r_lqdt = board_panel['Денежный рынок(LQDT)']
        fieldnames = [
            'start_date', 'end_date', 'window_length_days',
            'cagr', 'sortino', 'ulcer', 'max_drawdown', 'max_daily_drop',
            'asset_count', 'encoded_assets',
            'sortino_vs_1/N', 'sortino_vs_monkey', 'sortino_vs_markowitz', 'sortino_vs_cvar'
        ]
        srtn_delta = {}

        for strat in strategies:
            name = strat.name
            bench_key = name in ['Uniform 1/N', 'Random Monkey', 'Markowitz', 'Base CVaR']
            run = backtester.run_lumpsum_simulation(all_weights[name], initial_capital=1_000_000.0)
            result = run['Nominal_Capital']
            result_taxless = run['Nominal_Capital_Taxless']

            cagr = (result.iloc[-1] / result.iloc[0]) ** (252.0 / len(result)) - 1.0
            cagr_taxless = (result_taxless.iloc[-1] / result_taxless.iloc[0]) ** (252.0 / len(result)) - 1.0

            r_t = result_taxless.pct_change().dropna()

            excess_r = r_t - r_lqdt.loc[r_t.index]
            cagr_lqdt = np.exp(np.log1p(r_lqdt.loc[r_t.index]).sum()) ** (252.0 / len(result)) - 1.0
            cagr_excess = cagr_taxless - cagr_lqdt

            downside_r = np.minimum(0, excess_r)
            downside_std = np.sqrt(np.mean(downside_r ** 2))
            if downside_std > 0:
                if cagr_excess > 0: sortino = cagr_excess / (downside_std * np.sqrt(252))
                else: sortino = cagr_excess * downside_std * np.sqrt(252)
            else:
                sortino = -np.inf

            if bench_key: srtn_delta[name] = [r_t, cagr_taxless]
            else:
                sortino_bc_results = {}
                for bench in ['Uniform 1/N', 'Random Monkey', 'Markowitz', 'Base CVaR']:
                    excess_bc_r = r_t - srtn_delta[bench][0]
                    cagr_ex_bc = cagr_taxless - srtn_delta[bench][1]
                    downside_bc_r = np.minimum(0, excess_bc_r)
                    downside_std_bc = np.sqrt(np.mean(downside_bc_r ** 2))
                    sortino_bc_results[bench] = cagr_ex_bc / (max(downside_std_bc, 1e-9) * np.sqrt(252))

            max_drawdown = 0.0
            ulcer = 0.0
            cur_top = 1.0
            cur_ucr = 0.0
            for daily_inc in r_t:
                cur_top *= (1 + daily_inc)
                if cur_top >= 1.0:
                    cur_top = 1.0
                if 1.0 - cur_top > max_drawdown:
                    max_drawdown = 1.0 - cur_top
                cur_ucr += ((1 - cur_top) * 100) ** 2
            max_drawdown *= -100.0
            ulcer = -np.sqrt(cur_ucr / len(r_t))

            max_daily_drop = min(np.min(r_t) * 100.0, 0.0)

            metrics_dict = {'start_date': start, 'end_date': end, 'window_length_days': len(result),
            'cagr': cagr * 100, 'sortino': sortino, 'ulcer': ulcer, 'max_drawdown': max_drawdown, 'max_daily_drop': max_daily_drop,
            'asset_count': int(np.sum(random_bits) + 3), 'encoded_assets': encoded_assets}

            if not bench_key:
                metrics_dict['sortino_vs_1/N'] = sortino_bc_results['Uniform 1/N']
                metrics_dict['sortino_vs_monkey'] = sortino_bc_results['Random Monkey']
                metrics_dict['sortino_vs_markowitz'] = sortino_bc_results['Markowitz']
                metrics_dict['sortino_vs_cvar'] = sortino_bc_results['Base CVaR']
            else:
                metrics_dict['sortino_vs_1/N'] = np.nan
                metrics_dict['sortino_vs_monkey'] = np.nan
                metrics_dict['sortino_vs_markowitz'] = np.nan
                metrics_dict['sortino_vs_cvar'] = np.nan

            filepath = f'data/results/test_{name.replace('/', '_')}.csv'
            file_exists = os.path.isfile(filepath)

            with open(filepath, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(metrics_dict)

        print(f"Эксперимент {exper} завершен: {encoded_assets}")
    print("\n=== БЭКТЕСТ ЗАВЕРШЕН! ===")
