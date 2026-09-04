import pandas as pd
import warnings
import logging
from backtest_engine import PortfolioOrchestrator, PortfolioBacktester
from visualizer import PortfolioVisualizer
from benchmarks import (UniformStrategy, RandomMonkeyStrategy, StochasticMomentumStrategy, PersonalProfileStrategy, MarkowitzStrategy, BaseCVaRStrategy,
                        RobustParabolicCvarStrategy_old, RobustParabolicCvarStrategy_new, RobustParabolicCvarStrategy_medium)

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

if __name__ == "__main__":
    print("=== ЗАПУСК ГЛОБАЛЬНОГО ИСТОРИЧЕСКОГО БЭКТЕСТА СИСТЕМЫ ===")

    panel_path = 'data/matrix/global_board_panel.csv'
    vol_path = 'data/matrix/global_volatility_panel.csv'
    del_path = 'data/matrix/global_delist_panel.csv'

    board_panel = pd.read_csv(panel_path, parse_dates=['Date']).set_index('Date')
    volatility_panel = pd.read_csv(vol_path, parse_dates=['Date']).set_index('Date')
    delist_history = pd.read_csv(del_path, parse_dates=['Date']).set_index('Date')

    strategies = [
        UniformStrategy(), RandomMonkeyStrategy(), StochasticMomentumStrategy(), PersonalProfileStrategy(), MarkowitzStrategy(), BaseCVaRStrategy(),
        RobustParabolicCvarStrategy_old(), RobustParabolicCvarStrategy_new(), RobustParabolicCvarStrategy_medium()
    ]

    print("\n[ШАГ 1/3] Запуск динамической симуляции ребалансировок моделей...")
    orchestrator = PortfolioOrchestrator(board_panel, volatility_panel, strategies)
    all_weights = orchestrator.generate_weights_history(start_date="2015-01-01", end_date="2027-01-01")

    print("\n[ШАГ 2/3] Расчет накопления капитала по макро-сценариям...")
    backtester = PortfolioBacktester(board_panel, delist_history, inflation_annual=0.075, commission=0.0005)
    visualizer = PortfolioVisualizer()

    lump_results = {}
    dca_results = {}
    fire_results = {}

    for strat in strategies:
        name = strat.name
        logging.info(f"Прогон капитала для модели: {name}")

        lump_results[name] = backtester.run_lumpsum_simulation(all_weights[name], initial_capital=1_000_000.0)
        dca_results[name] = backtester.run_dca_simulation(all_weights[name], extra_capital=50_000.0)
        fire_results[name] = backtester.run_fire_simulation(all_weights[name], initial_capital=6_000_000.0, wherewithal=60_000.0)

    print("\n[ШАГ 3/3] Генерация и визуализация сравнительных графиков...")
    lqdt_series = board_panel['Денежный рынок(LQDT)']
    visualizer.plot_scenario_comparison(lump_results, "lumpsum", lqdt_series)
    visualizer.plot_scenario_comparison(dca_results, "dca", lqdt_series)
    visualizer.plot_scenario_comparison(fire_results, "fire", lqdt_series)

    print("\n=== БЭКТЕСТ ЗАВЕРШЕН! ===")