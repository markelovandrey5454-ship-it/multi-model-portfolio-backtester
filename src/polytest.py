import pandas as pd
from datetime import datetime
import random
import numpy as np
import logging
from backtest_engine import PortfolioOB
import os
import csv
from benchmarks import (UniformStrategy, RandomMonkeyStrategy, MarkowitzStrategy, BaseCVaRStrategy, StochasticMomentumStrategy,
                        ML_RankEnsemble_Frankenstein, RobustParabolicCvarStrategy_medium)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
BASE32_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUV"
n = 250
benchmark = ['Uniform 1/N', 'Random Monkey', 'Markowitz', 'Base CVaR', 'Stochastic Momentum']

if __name__ == "__main__":
    board_panel = pd.read_csv('data/matrix/global_board_panel.csv', parse_dates=['Date']).set_index('Date')
    volatility_panel = pd.read_csv('data/matrix/global_volatility_panel.csv', parse_dates=['Date']).set_index('Date')
    volume_panel = pd.read_csv('data/matrix/global_volume_panel.csv', parse_dates=['Date']).set_index('Date')
    delist_history = pd.read_csv('data/matrix/global_delist_panel.csv', parse_dates=['Date']).set_index('Date')

    r_lqdt = board_panel['Денежный рынок(LQDT)']
    r_mcftrr = pd.read_csv('data/cache/IMOEX ПД + налог РФ.csv', parse_dates=['Date']).set_index('Date')['IMOEX ПД + налог РФ_close'].reindex(r_lqdt.index).ffill().pct_change().dropna()

    fieldnames = [
        'start_date', 'end_date', 'window_length_days',
        'cagr', 'sortino', 'jenser',
        'ulcer', 'max_drawdown', 'max_daily_drop',
        'asset_count', 'encoded_assets'
    ]

    print("=== ЗАПУСК СЛУЧАЙНОГО БЭКТЕСТА СИСТЕМЫ ===")
    for exper in range(1, n + 1):
        delt_date = random.randint(3,5)
        start_date = random.randint(int(datetime(2015, 1, 1).timestamp()), int(datetime(2026 - delt_date, 10, 1).timestamp()))
        start_date = pd.to_datetime(datetime.fromtimestamp(start_date).date())
        end_date = start_date + pd.DateOffset(years=delt_date)
        start = start_date.strftime("%Y-%m-%d")
        end = end_date.strftime("%Y-%m-%d")

        final_bit_mask = np.zeros(len(volatility_panel.columns), dtype=int)
        alive_assets_mask = (~np.isnan(volatility_panel.loc[start:end])).sum(axis=0) != 0
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

        strategies = [
            UniformStrategy(),
            RandomMonkeyStrategy(),
            StochasticMomentumStrategy(),
            BaseCVaRStrategy(),
            MarkowitzStrategy(),
            RobustParabolicCvarStrategy_medium(),
            ML_RankEnsemble_Frankenstein()
        ]

        ob = PortfolioOB(
            board_panel=board_panel,
            volatility_panel=volatility_panel,
            volume_panel=volume_panel,
            delist_history=delist_history,
            strategies=strategies,
            commission=0.0005
        )

        results_df = ob.simulation(
            start_date=start,
            end_date=end,
            initial_capital=1_000_000.0,
            key=True,
            assets=assets_str
        )

        n_days = len(results_df)
        welch = {}

        for strat in strategies:
            name = strat.name
            bench_key = name in benchmark
            result = results_df[f'{name}_Lump']
            result_taxless = results_df[f'{name}_Lump_tf']

            cagr = (result.iloc[-1] / result.iloc[0]) ** (252.0 / len(result)) - 1.0
            cagr_taxless = (result_taxless.iloc[-1] / result_taxless.iloc[0]) ** (252.0 / len(result)) - 1.0

            r_t = result_taxless.pct_change().dropna()

            excess_r = r_t - r_lqdt.loc[r_t.index]
            excess_moex = r_mcftrr.loc[r_t.index] - r_lqdt.loc[r_t.index]

            cov_matrix = np.cov(excess_r, excess_moex)
            beta = cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-12)

            cagr_lqdt = np.exp(np.log1p(r_lqdt.loc[r_t.index]).sum()) ** (252.0 / len(result)) - 1.0
            cagr_moex = np.exp(np.log1p(r_mcftrr.loc[r_t.index]).sum()) ** (252.0 / len(result)) - 1.0
            cagr_excess = cagr_taxless - cagr_lqdt

            jenser = cagr_taxless - cagr_lqdt - beta * (cagr_moex - cagr_lqdt)

            downside_r = np.minimum(0, excess_r)
            downside_std = np.sqrt(np.mean(downside_r ** 2))
            if downside_std > 0:
                if cagr_excess > 0: sortino = cagr_excess / (downside_std * np.sqrt(252))
                else: sortino = cagr_excess * downside_std * np.sqrt(252)
            else: sortino = -np.inf

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
            'cagr': cagr * 100, 'sortino': sortino, 'jenser' : jenser * 100,
            'ulcer': ulcer, 'max_drawdown': max_drawdown, 'max_daily_drop': max_daily_drop,
            'asset_count': int(np.sum(random_bits) + 3), 'encoded_assets': encoded_assets}

            filepath = f'data/results/test_{name.replace('/', '_')}.csv'
            file_exists = os.path.isfile(filepath)

            with open(filepath, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(metrics_dict)

        print(f"Эксперимент {exper} завершен: {encoded_assets}")
    print("\n=== БЭКТЕСТ ЗАВЕРШЕН! ===")
