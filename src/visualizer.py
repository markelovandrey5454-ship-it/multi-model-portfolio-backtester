import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy import stats

STRATEGY_COLORS = {
    'Uniform 1/N': '#9E9E9E',
    'Random Monkey': '#E91E63',
    'Stochastic Momentum': '#2196F3',
    'Markowitz': '#9C27B0',
    'Base CVaR': '#FF9800',
    '0.1 prototype': '#4CAF50',
    'RankEnsemble_Frankenstein': '#FF5722'
}

target_strategies = ['0.1 prototype', 'RankEnsemble_Frankenstein']


class PortfolioVisualizer:
    def __init__(self, output_dir: str = "data/results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        plt.style.use(swg if (swg := 'seaborn-v0_8-whitegrid') in plt.style.available else 'default')

    def plot_scenario_comparison(self, all_strategies_results: dict, scenario_name: str, r_lqdt: pd.Series = None):
        is_lumpsum = scenario_name.lower() == 'lumpsum'

        if is_lumpsum: fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [7, 3]})
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(14, 7))
            ax2 = None

        inflation_plotted = False

        for strat_name, df_res in all_strategies_results.items():
            color = STRATEGY_COLORS.get(strat_name)
            alpha = 1.0 if strat_name in target_strategies else 0.35

            if not inflation_plotted:
                ax1.plot(df_res.index, df_res['Inflation_Benchmark'], label=f"Инфляционный Бенчмарк (Финальный капитал: {df_res['Inflation_Benchmark'].iloc[-1]:,.0f} руб)", color='black', linestyle='--', linewidth=2.5)
                inflation_plotted = True

            if is_lumpsum:
                cagr = (df_res['Nominal_Capital'].iloc[-1] / df_res['Nominal_Capital'].iloc[0]) ** (252.0 / len(df_res)) - 1.0
                cagr_taxless = (df_res['Nominal_Capital_Taxless'].iloc[-1] / df_res['Nominal_Capital_Taxless'].iloc[0]) ** (252.0 / len(df_res)) - 1.0

                r_t = df_res['Nominal_Capital_Taxless'].pct_change().dropna()
                if r_lqdt is not None:
                    excess_r = r_t - r_lqdt.loc[r_t.index]
                    cagr_lqdt = np.exp(np.log1p(r_lqdt.loc[r_t.index]).sum()) ** (252.0 / len(df_res)) - 1.0
                    cagr_excess = cagr_taxless - cagr_lqdt
                else:
                    excess_r = r_t
                    cagr_excess = cagr_taxless
                downside_r = np.minimum(0, excess_r)
                downside_std = np.sqrt(np.mean(downside_r ** 2))
                if downside_std > 0: sortino = (cagr_excess / downside_std) / np.sqrt(252)
                else: sortino = 0

                max_drawdown = 0.0
                cur_top = 1.0
                for daily_inc in r_t:
                    cur_top *= 1 + daily_inc
                    if cur_top >= 1.0:
                        cur_top = 1.0
                    elif 1.0 - cur_top > max_drawdown:
                        max_drawdown = 1.0 - cur_top

                label_str = (f"{strat_name} (CAGR: {cagr * 100:.2f}%, Sortino: {sortino:.2f}, Calmar: {cagr / max_drawdown:.2f})")
            else:
                final_cap = df_res['Nominal_Capital'].iloc[-1]
                label_str = f"{strat_name} (Финальный капитал: {final_cap:,.0f} руб)"

            line, = ax1.plot(df_res.index, df_res['Nominal_Capital'], label=label_str, linewidth=2, color=color, alpha=alpha)

            if is_lumpsum and ax2 is not None:
                scatter_alpha = 0.35 if alpha == 1.0 else 0.1
                ax2.scatter(r_t.index, r_t.values, alpha=scatter_alpha, s=8, color=color)

        ax1.set_title(f"СРАВНИТЕЛЬНОЕ СОРЕВНОВАНИЕ МОДЕЛЕЙ ОПТИМИЗАЦИИ — СЦЕНАРИЙ {scenario_name.upper()}", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Капитал (руб)", fontsize=12)
        ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        ax1.legend(loc="upper left", fontsize=11, frameon=True)

        if is_lumpsum and ax2 is not None:
            ax2.set_xlabel("Дата", fontsize=12)
            ax2.set_ylabel("Дневная доходность", fontsize=12)
            ax2.set_yscale('symlog', linthresh=0.01)
            ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
            ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        else: ax1.set_xlabel("Дата", fontsize=12)

        plt.tight_layout()
        file_path = os.path.join(self.output_dir, f"comparison_{scenario_name.lower()}.png")
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f"График сценария {scenario_name} успешно сохранен по пути: {file_path}")


class TRVisualizer:
    def __init__(self, output_dir: str = "data/results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        plt.style.use(swg if (swg := 'seaborn-v0_8-whitegrid') in plt.style.available else 'default')

    def plot_tr_shares(self, shares: dict, sv_sh: bool, lag: bool):
        pass
