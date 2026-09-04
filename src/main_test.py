import os
import pandas as pd
from visualizer import DatatestVisualizer


ASSET_NAMES = pd.read_csv('data/matrix/global_volatility_panel.csv', nrows=0).drop(columns=['Date'], errors='ignore').columns.tolist()
total_assets_count = len(ASSET_NAMES)
board_panel = pd.read_csv('data/matrix/global_board_panel.csv', parse_dates=['Date']).set_index('Date')

if __name__ == "__main__":
    data_dir = 'data/results'
    visualizer = DatatestVisualizer()

    strategy_files = {
        'Base CVaR': 'test_Base CVaR.csv',
        'Robust CVaR (New)': 'test_Controlled-risk prototype.csv',
        'Robust CVaR (Old)': 'test_High-risk prototype.csv',
        'Robust CVaR (0.1)': 'test_0.1 prototype.csv',
        'Uniform 1/N': 'test_Uniform 1_N.csv',
        'Markowitz': 'test_Markowitz.csv',
        'Random Monkey': 'test_Random Monkey.csv'
    }

    raw_dfs = {}
    for label, fname in strategy_files.items():
        path = os.path.join(data_dir, fname)
        raw_dfs[label] = pd.read_csv(path)

    print(f"Данные успешно считаны. Найдено {len(raw_dfs['Robust CVaR (New)'])} прогонов симуляции.")

    all_results, worst_results = visualizer.generate_research_plots(raw_dfs)

    for model_name in ['Robust CVaR (Old)', 'Robust CVaR (New)', 'Robust CVaR (0.1)']:
        df_bad = worst_results[model_name]
        df_all = all_results[model_name]

        print(f"\n=== ИССЛЕДОВАНИЕ КРИТИЧЕСКИХ ПРОВАЛОВ ДЛЯ: {model_name.upper()} ===")

        if df_bad.empty:
            print("Модель показала абсолютную устойчивость во всех окнах!")
            continue
        else: print(f"Выявлено значимых аномальных проработок: {len(df_bad)} шт.")
