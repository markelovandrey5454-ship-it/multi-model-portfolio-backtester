from backtest_engine import PortfolioOB
from visualizer import PortfolioVisualizer
from benchmarks import (UniformStrategy, RandomMonkeyStrategy, StochasticMomentumStrategy, MarkowitzStrategy, BaseCVaRStrategy,
                        RobustParabolicCvarStrategy_old, RobustParabolicCvarStrategy_new, RobustParabolicCvarStrategy_medium,
                        ML_RankEnsemble_Frankenstein)

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
key = False
initial_capital = 100_000_000.0 if key else 1_000_000.0

if __name__ == "__main__":
    print("=== ЗАПУСК ОБЪЕДИНЕННОГО ВЕКТОРИЗИРОВАННОГО БЭКТЕСТА ===")

    board_panel = pd.read_csv('data/matrix/global_board_panel.csv', parse_dates=['Date']).set_index('Date')
    volatility_panel = pd.read_csv('data/matrix/global_volatility_panel.csv', parse_dates=['Date']).set_index('Date')
    volume_panel = pd.read_csv('data/matrix/global_volume_panel.csv', parse_dates=['Date']).set_index('Date')
    delist_history = pd.read_csv('data/matrix/global_delist_panel.csv', parse_dates=['Date']).set_index('Date')

    strategies = [
        UniformStrategy(),
        RandomMonkeyStrategy(),
        StochasticMomentumStrategy(),
        BaseCVaRStrategy(),
        MarkowitzStrategy(),
        #RobustParabolicCvarStrategy_old(),
        #RobustParabolicCvarStrategy_new(),
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
        start_date="2015-01-01",
        end_date="2027-12-31",
        initial_capital=initial_capital,
        key=key,
        #assets="111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111"
    )

    visualizer = PortfolioVisualizer()

    lump_results = {}
    dca_results = {}
    fire_results = {}

    for strat in strategies:
        name = strat.name
        logging.info(f"Сборка выходных панелей для визуализатора по модели: {name}")

        lump_results[name] = pd.DataFrame(index=results_df.index)
        lump_results[name]['Nominal_Capital'] = results_df[f'{name}_Lump']
        lump_results[name]['Nominal_Capital_Taxless'] = results_df[f'{name}_Lump_tf']
        lump_results[name]['Inflation_Benchmark'] = results_df['Inflation_Benchmark_Lump']

        if not key:
            dca_results[name] = pd.DataFrame(index=results_df.index)
            dca_results[name]['Nominal_Capital'] = results_df[f'{name}_DCA']
            dca_results[name]['Inflation_Benchmark'] = results_df['Inflation_Benchmark_DCA']

            fire_results[name] = pd.DataFrame(index=results_df.index)
            fire_results[name]['Nominal_Capital'] = results_df[f'{name}_FIRE']
            fire_results[name]['Inflation_Benchmark'] = results_df['Inflation_Benchmark_FIRE']

    lqdt_series = board_panel['Денежный рынок(LQDT)']

    visualizer.plot_scenario_comparison(initial_capital, lump_results, "lumpsum", lqdt_series)
    if not key:
        visualizer.plot_scenario_comparison(50_000, dca_results, "dca", lqdt_series)
        visualizer.plot_scenario_comparison(6_000_000, fire_results, "fire", lqdt_series)

        results_df.to_csv('data/results/strat_csv/human.csv')
    else: results_df.to_csv('data/results/strat_csv/fond.csv')

    print("\n=== ГЛОБАЛЬНЫЙ БЭКТЕСТ И СБОРКА ГРАФИКОВ ЗАВЕРШЕНЫ УСПЕШНО! ===")
