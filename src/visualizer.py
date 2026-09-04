import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os
import numpy as np
import pandas as pd


class PortfolioVisualizer:
    def __init__(self, output_dir: str = "data/results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        plt.style.use(swg if (swg := 'seaborn-v0_8-whitegrid') in plt.style.available else 'default')

    def plot_scenario_comparison(self, all_strategies_results: dict, scenario_name: str, r_lqdt: pd.Series = None):
        is_lumpsum = scenario_name.lower() == 'lumpsum'

        if is_lumpsum:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [7, 3]})
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(14, 7))
            ax2 = None

        inflation_plotted = False

        for strat_name, df_res in all_strategies_results.items():
            if df_res.empty:
                continue
            if not inflation_plotted:
                ax1.plot(df_res.index, df_res['Inflation_Benchmark'],
                         label="Инфляционный Бенчмарк", color='black', linestyle='--', linewidth=2.5)
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
                if downside_std > 0:
                    sortino = (cagr_excess / downside_std) / np.sqrt(252)
                else: sortino = 0

                max_drawdown = 0.0
                cur_top = 1.0
                for daily_inc in r_t:
                    cur_top *= 1 + daily_inc
                    if cur_top >= 1.0:
                        cur_top = 1.0
                    elif 1.0 - cur_top > max_drawdown:
                        max_drawdown = 1.0 - cur_top

                label_str = f"{strat_name} (CAGR: {cagr * 100:.2f}%, Sortino: {sortino:.2f}, Calmar: {cagr / max_drawdown:.2f})"
            else:
                final_cap = df_res['Nominal_Capital'].iloc[-1]
                label_str = f"{strat_name} (Финальный капитал: {final_cap:,.0f} руб)"

            line, = ax1.plot(df_res.index, df_res['Nominal_Capital'], label=label_str, linewidth=2)

            if is_lumpsum and ax2 is not None:
                ax2.scatter(r_t.index, r_t.values, alpha=0.5, s=8, color=line.get_color())

        ax1.set_title(f"СРАВНИТЕЛЬНОЕ СОРЕВНОВАНИЕ МОДЕЛЕЙ ОПТИМИЗАЦИИ — СЦЕНАРИЙ {scenario_name.upper()}",
                                                                                fontsize=14, fontweight='bold')

        ax1.set_ylabel("Капитал (руб)", fontsize=12)
        ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        ax1.legend(loc="upper left", fontsize=11, frameon=True)

        if is_lumpsum and ax2 is not None:
            ax2.set_xlabel("Дата", fontsize=12)
            ax2.set_ylabel("Дневная доходность", fontsize=12)
            ax2.set_yscale('symlog', linthresh=0.01)
            ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
            ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        else:
            ax1.set_xlabel("Дата", fontsize=12)

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


class DatatestVisualizer:
    def __init__(self, output_dir: str = "data/results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        plt.style.use(swg if (swg := 'seaborn-v0_8-whitegrid') in plt.style.available else 'default')

    def generate_research_plots(self, m_data):
        benchmarks_keys = ['Uniform 1/N', 'Random Monkey', 'Markowitz', 'Base CVaR']
        grail_keys = [k for k in m_data.keys() if k not in benchmarks_keys]
        old_key = grail_keys[0]
        new_key = grail_keys[1]
        medium_key = grail_keys[2]
        fig, axes = plt.subplots(1, 4, figsize=(36, 6))

        boxplot_data = []
        for label, df in m_data.items():
            boxplot_data.append(pd.DataFrame({'Value': df['cagr'], 'Strategy': label}))
        combined_box_df = pd.concat(boxplot_data, ignore_index=True)

        sns.boxplot(data=combined_box_df, x='Strategy', y='Value', ax=axes[0], palette='Set2', hue='Strategy', legend=False)
        axes[0].set_title('Абсолютная скорость роста портфелей (CAGR, %)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Проценты годовых')
        axes[0].tick_params(axis='x', rotation=25)
        axes[0].grid(True, alpha=0.3)

        alpha_old_data = []
        df_old = m_data[old_key]
        for bench in benchmarks_keys:
            delta_cagr = df_old['cagr'].values - m_data[bench]['cagr'].values
            alpha_old_data.append(pd.DataFrame({'Delta_CAGR': delta_cagr, 'Benchmark': bench}))

        combined_old_df = pd.concat(alpha_old_data, ignore_index=True)
        sns.boxplot(data=combined_old_df, x='Benchmark', y='Delta_CAGR', ax=axes[1], palette='Pastel1', hue='Benchmark', legend=False)
        axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5, label='Паритет (Ничья)')
        axes[1].set_title(f'Чистая Альфа НОВОЙ модели\n(Δ CAGR для {old_key}, %)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Разность CAGR')
        axes[1].set_xlabel('Бенчмарк')
        axes[1].tick_params(axis='x', rotation=30)
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)

        alpha_new_data = []
        df_new = m_data[new_key]
        for bench in benchmarks_keys:
            delta_cagr = df_new['cagr'].values - m_data[bench]['cagr'].values
            alpha_new_data.append(pd.DataFrame({'Delta_CAGR': delta_cagr, 'Benchmark': bench}))

        combined_new_df = pd.concat(alpha_new_data, ignore_index=True)
        sns.boxplot(data=combined_new_df, x='Benchmark', y='Delta_CAGR', ax=axes[2], palette='Pastel1', hue='Benchmark', legend=False)
        axes[2].axhline(0, color='red', linestyle='--', linewidth=1.5, label='Паритет (Ничья)')
        axes[2].set_title(f'Чистая Альфа СТАРОЙ модели\n(Δ CAGR для {new_key}, %)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Разность CAGR')
        axes[2].set_xlabel('Бенчмарк')
        axes[2].tick_params(axis='x', rotation=30)
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

        alpha_medium_data = []
        df_medium = m_data[medium_key]
        for bench in benchmarks_keys:
            delta_cagr = df_medium['cagr'].values - m_data[bench]['cagr'].values
            alpha_medium_data.append(pd.DataFrame({'Delta_CAGR': delta_cagr, 'Benchmark': bench}))

        combined_medium_df = pd.concat(alpha_medium_data, ignore_index=True)
        sns.boxplot(data=combined_medium_df, x='Benchmark', y='Delta_CAGR', ax=axes[3], palette='Pastel1', hue='Benchmark', legend=False)
        axes[3].axhline(0, color='red', linestyle='--', linewidth=1.5, label='Паритет (Ничья)')
        axes[3].set_title(f'Чистая Альфа 0.1 модели\n(Δ CAGR для {medium_key}, %)', fontsize=12, fontweight='bold')
        axes[3].set_ylabel('Разность CAGR')
        axes[3].set_xlabel('Бенчмарк')
        axes[3].tick_params(axis='x', rotation=30)
        axes[3].legend(loc='upper right')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '01_alpha_and_cagr_comparison.png'), dpi=300)
        plt.close()

        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        for idx, (label, df) in enumerate(m_data.items()):
            sns.kdeplot(data=df['max_drawdown'], ax=axes[0], label=label, fill=True, alpha=0.08, clip=(None, 0.0), cut=0.0)
            sns.kdeplot(data=df['max_daily_drop'], ax=axes[1], label=label, fill=True, alpha=0.08, clip=(None, 0.0), cut=0.0)
            sns.kdeplot(data=df['ulcer'], ax=axes[2], label=label, fill=True, alpha=0.08, clip=(None, 0.0), cut=0.0)

        axes[0].set_title('Плотность распределения Max Drawdown (Пиковый шок)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Худшая историческая просадка портфеля (%)')
        axes[0].set_ylabel('Плотность вероятности')
        axes[0].grid(True, alpha=0.2)
        axes[0].legend(loc='upper left')
        axes[1].set_title('Плотность Max Daily Drop (Хвостовой риск сессии)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Максимальный дневной убыток портфеля (%)')
        axes[1].set_ylabel('Плотность вероятности')
        axes[1].grid(True, alpha=0.2)
        axes[1].legend(loc='upper left')
        axes[2].set_title('Плотность Индекса Ульцера (Временной износ капитала)', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Среднеквадратичная глубина и длительность просадок (%)')
        axes[2].set_ylabel('Плотность вероятности')
        axes[2].grid(True, alpha=0.2)
        axes[2].legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '02_simple_comparison.png'), dpi=300)
        plt.close()

        colors = {'Победа': '#4CAF50', 'Ничья': '#9E9E9E', 'Поражение': '#F44336'}
        worst_scenarios_dict = {}
        all_scenarios_dict = {}

        fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharey=True)

        for idx, grail in enumerate([old_key, new_key, medium_key]):
            df_grail = m_data[grail]
            n_runs = len(df_grail)
            matrix_statuses = []

            extended_x_labels = benchmarks_keys + ['1 vs ALL (Maximin)']
            plot_proportions = {label: {'Победа': 0, 'Ничья': 0, 'Поражение': 0} for label in extended_x_labels}

            for bench in benchmarks_keys:
                col_mapping = {
                    'Uniform 1/N': 'sortino_vs_1/N',
                    'Random Monkey': 'sortino_vs_monkey',
                    'Markowitz': 'sortino_vs_markowitz',
                    'Base CVaR': 'sortino_vs_cvar'
                }
                col_name = col_mapping[bench]
                sortino_annual = df_grail[col_name].values
                n_days = df_grail['window_length_days'].values
                t_stats = sortino_annual * np.sqrt(n_days / 252.0)

                print(f"\nTarget: {grail} vs {bench}")
                print(f"  * Максимальный t-балл (потенциал победы): {np.nanmax(t_stats):.4f}")
                print(f"  * Минимальный t-балл (риск провала): {np.nanmin(t_stats):.4f}")
                print(f"  * Средний t-балл выборки: {np.nanmean(t_stats):.4f}")

                statuses = []
                for t in t_stats:
                    if t >= 1.65:
                        statuses.append('Победа')
                    elif -2.33 <= t <= 0.0:
                        statuses.append('Поражение')
                    else:
                        statuses.append('Ничья')
                statuses = np.array(statuses)
                matrix_statuses.append(statuses)

                for status in ['Победа', 'Ничья', 'Поражение']:
                    count = np.sum(statuses == status)
                    plot_proportions[bench][status] = (count / n_runs) * 100.0

                for status in ['Победа', 'Ничья', 'Поражение']:
                    mask = (statuses == status)
                    matches = np.sum(mask)
                    if matches > 0:
                        avg_cagr_g = df_grail.loc[mask, 'cagr'].mean()
                        avg_cagr_b = m_data[bench].loc[mask, 'cagr'].mean()
                        avg_ulcer_g = df_grail.loc[mask, 'ulcer'].mean()
                        avg_ulcer_b = m_data[bench].loc[mask, 'ulcer'].mean()

                        print(f"  ↳ Зона [{status}] ({matches}/{n_runs} прогонов, {(matches / n_runs) * 100:.1f}%):")
                        print(f"     • Средний CAGR: Грааль = {avg_cagr_g:.2f}%, Бенчмарк = {avg_cagr_b:.2f}% (Δ = {avg_cagr_g - avg_cagr_b:.2f}%)")
                        print(f"     • Средний Ulcer Index: Грааль = {avg_ulcer_g:.2f}%, Бенчмарк = {avg_ulcer_b:.2f}%")
                    else: print(f"  ↳ Зона [{status}]: 0 прогонов (модель не попадала в этот сценарий)")

            matrix_statuses = np.array(matrix_statuses).T
            maximin_statuses = []

            for run_idx in range(n_runs):
                row = matrix_statuses[run_idx]
                if 'Поражение' in row: maximin_statuses.append('Поражение')
                elif 'Ничья' in row: maximin_statuses.append('Ничья')
                else: maximin_statuses.append('Победа')

            maximin_statuses = np.array(maximin_statuses)

            for status in ['Победа', 'Ничья', 'Поражение']:
                count = np.sum(maximin_statuses == status)
                plot_proportions['1 vs ALL (Maximin)'][status] = (count / n_runs) * 100.0

            df_grail['final_maximin_status'] = maximin_statuses
            all_scenarios_dict[grail] = df_grail.copy()
            worst_scenarios_dict[grail] = df_grail[maximin_statuses == 'Поражение'].copy()

            ax = axes[idx]
            bottoms = np.zeros(len(extended_x_labels))

            for status in ['Поражение', 'Ничья', 'Победа']:
                heights = [plot_proportions[label][status] for label in extended_x_labels]
                ax.bar(extended_x_labels, heights, bottom=bottoms, label=status, color=colors[status], alpha=0.85, width=0.55)
                bottoms += heights

            ax.set_title(f"Распределение зон устойчивости\nдля {grail} (с учетом 1 vs ALL)", fontsize=12, fontweight='bold')
            ax.set_xlabel('Конкурирующий режим')
            ax.tick_params(axis='x', rotation=25)
            ax.set_ylim(0, 100)
            ax.grid(True, axis='y', alpha=0.3)
            if idx == 0:
                ax.set_ylabel('Процент от общего числа прогонов (%)')
                ax.legend(loc='lower left', title="Статус прогона")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '03_multi-comparison.png'), dpi=300)
        plt.close()
        return all_scenarios_dict, worst_scenarios_dict
