import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from visualizer import DatatestVisualizer
from empty_box import decode_portfolio_to_bits


ASSET_NAMES = pd.read_csv('data/matrix/global_volatility_panel.csv', nrows=0).drop(columns=['Date'], errors='ignore').columns.tolist()
total_assets_count = len(ASSET_NAMES)
board_panel = pd.read_csv('data/matrix/global_board_panel.csv', parse_dates=['Date']).set_index('Date')

if __name__ == "__main__":
    data_dir = 'data/results'
    visualizer = DatatestVisualizer()

    strategy_files = {
        '0.1 prototype': 'test_0.1 prototype.csv',
        'RankEnsemble_Frankenstein': 'test_RankEnsemble_Frankenstein.csv',
        'Uniform 1/N': 'test_Uniform 1_N.csv',
        'Base CVaR': 'test_Base CVaR.csv',
        'Markowitz': 'test_Markowitz.csv',
        'Random Monkey': 'test_Random Monkey.csv',
        'Stochastic Momentum': 'test_Stochastic Momentum.csv'
    }

    raw_dfs = {}
    for label, fname in strategy_files.items():
        path = os.path.join(data_dir, fname)
        raw_dfs[label] = pd.read_csv(path)

    print(f"Данные успешно считаны. Найдено {len(raw_dfs['0.1 prototype'])} прогонов симуляции.")

    visualizer.generate_research_plots(raw_dfs)
