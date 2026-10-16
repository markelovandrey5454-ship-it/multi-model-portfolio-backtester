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


class DatatestVisualizer:
    def __init__(self, output_dir: str = "data/results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        plt.style.use(swg if (swg := 'seaborn-v0_8-whitegrid') in plt.style.available else 'default')

    def _plot_confidence_ellipse(self, x, y, ax, color, n_std=1.96):
        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1] + 1e-12)

        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)

        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=color, alpha=0.15, edgecolor=color, linestyle='--', linewidth=1.5)

        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)

        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)

    def _get_dynamic_delta(self, start, end, lqdt_cum_index):
        cagr_lqdt = np.zeros(n := len(start))
        for i in range(n):
            idx_start = lqdt_cum_index.index.get_indexer([pd.to_datetime(start[i])], method='pad')[0]
            idx_end = lqdt_cum_index.index.get_indexer([pd.to_datetime(end[i])], method='pad')[0]

            val_start = lqdt_cum_index.iloc[idx_start]
            val_end = lqdt_cum_index.iloc[idx_end]

            cagr_lqdt[i] = (val_end / val_start) ** (252.0 / (idx_end - idx_start)) - 1.0
        return cagr_lqdt * 100.0

    def generate_research_plots(self, m_data):
        benchmarks_keys = ['Uniform 1/N', 'Random Monkey', 'Markowitz', 'Base CVaR', 'Stochastic Momentum']
        grail_keys = [k for k in m_data.keys() if k not in benchmarks_keys]

        description = {
            'cagr': {'title': 'Абсолютная скорость роста портфелей (CAGR)',
                     'ylabel': 'Проценты годовых (%)'},
            'sortino': {'title': 'Коэффициент Сортино (Sortino Ratio)',
                     'ylabel': 'Значение коэффициента'},
            'jenser': {'title': 'Чистая годовая Альфа Дженсена (Jensen\'s Alpha)',
                     'ylabel': 'Избыточная доходность (% годовых)'}
        }

        fig, axes = plt.subplots(1, (n:=len(description)), figsize=(8*n, 6))

        for i, io in enumerate(['cagr', 'sortino', 'jenser']):
            boxplot_data = []
            for label, df in m_data.items(): boxplot_data.append(pd.DataFrame({'Value': df[io], 'Strategy': label}))
            combined_box = pd.concat(boxplot_data, ignore_index=True)

            sns.boxplot(data=combined_box, x='Strategy', y='Value', ax=axes[i], palette=STRATEGY_COLORS, hue='Strategy', legend=False)
            if io == 'sortino':

                def forward_scale(y):
                    return np.where(y >= 0, y, y * 30)

                def inverse_scale(y):
                    return np.where(y >= 0, y, y / 30)

                axes[i].set_yscale('function', functions=(forward_scale, inverse_scale))
                axes[i].set_yticks([-0.05, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
                axes[i].set_ylim(bottom=-0.1)
            axes[i].set_title(description[io]['title'], fontsize=12, fontweight='bold')
            axes[i].set_ylabel(description[io]['ylabel'])
            axes[i].tick_params(axis='x', rotation=25)
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '01_yield_comparison.png'), dpi=300)
        plt.close()

        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        for idx, (label, df) in enumerate(m_data.items()):
            #if label in benchmarks_keys: continue
            color = STRATEGY_COLORS.get(label)
            sns.kdeplot(data=df['max_drawdown'], ax=axes[0], label=label, fill=True, alpha=0.08, clip=(None, 0.0), cut=0.0, color=color)
            sns.kdeplot(data=df['max_daily_drop'], ax=axes[1], label=label, fill=True, alpha=0.08, clip=(None, 0.0), cut=0.0, color=color)
            sns.kdeplot(data=df['ulcer'], ax=axes[2], label=label, fill=True, alpha=0.08, clip=(None, 0.0), cut=0.0, color=color)

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
        plt.savefig(os.path.join(self.output_dir, '02_risk_comparison.png'), dpi=300)
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(24, 6))

        df_1 = m_data['Uniform 1/N']
        df_2 = m_data['Random Monkey']
        starts = df_1['start_date'].values
        ends = df_1['end_date'].values
        dict_4 = {}

        lqdt_panel = pd.read_csv('data/matrix/global_board_panel.csv', parse_dates=['Date']).set_index('Date')['Денежный рынок(LQDT)']
        lqdt_cum_index = np.exp(np.log1p(lqdt_panel).cumsum())

        thresh = {}
        for idx, (alpha, risk) in enumerate({'cagr': 'max_drawdown', 'jenser': 'ulcer'}.items()):
            delta = 0.0 if alpha == 'jenser' else (delta_cagr_lqdt := self._get_dynamic_delta(starts, ends, lqdt_cum_index))
            k_1 = (df_1[alpha].values - delta) / (df_1[risk].values ** 2 + 1e-12)
            k_2 = (df_2[alpha].values - delta) / (df_2[risk].values ** 2 + 1e-12)
            delta_k_noise = np.concat([k_1 - k_2, k_2 - k_1])

            thresh[idx] = {'high': np.percentile(delta_k_noise, 99.9), 'low': np.percentile(delta_k_noise, 75)}

        for label, df in m_data.items():
            #if label in benchmarks_keys: continue
            strat_color = STRATEGY_COLORS.get(label)
            x_val = np.abs(df['max_drawdown'].values)
            y_val = df['cagr'].values

            if label in grail_keys:
                dict_4[label] = {}
                for bench in benchmarks_keys:
                    k_m = (y_val - delta_cagr_lqdt) / (x_val ** 2 + 1e-12)
                    k_b = (m_data[bench]['cagr'].values - delta_cagr_lqdt) / (np.abs(m_data[bench]['max_drawdown'].values) ** 2 + 1e-12)
                    delta_k = k_m - k_b

                    statuses = np.where(delta_k >= thresh[0]['high'], 3,
                                        np.where(delta_k > 0, 2,
                                        np.where(delta_k >= thresh[0]['low'], 1, 0)))
                    dict_4[label][bench] = {'statuses_0': statuses}

                dict_4[label]['1vsAll'] = {'statuses_0': np.min(np.array(list(dict_4[label][bench]['statuses_0'] for bench in benchmarks_keys)), axis=0)}

            axes[0].scatter(x_val, y_val, alpha=0.5, s=20, color=strat_color, label=label)
            self._plot_confidence_ellipse(x_val, y_val, axes[0], strat_color)

            x_val = np.abs(df['ulcer'].values)
            y_val = df['jenser'].values

            if label in grail_keys:
                for bench in benchmarks_keys:
                    k_m = y_val / (x_val ** 2 + 1e-12)
                    k_b = m_data[bench]['jenser'].values / (np.abs(m_data[bench]['ulcer'].values) ** 2 + 1e-12)
                    delta_k = k_m - k_b

                    statuses = np.where(delta_k >= thresh[1]['high'], 3,
                                        np.where(delta_k >= thresh[1]['low'], 2,
                                        np.where(delta_k >= -thresh[1]['low'], 1, 0)))
                    dict_4[label][bench]['statuses_1'] = statuses

                    res = stats.wilcoxon(df['sortino'].values, m_data[bench]['sortino'].values, alternative='greater')
                    dict_4[label][bench]['wilcoxon'] = (1.0 - res.pvalue) * 100.0

                dict_4[label]['1vsAll']['statuses_1'] = np.min(np.array(list(dict_4[label][bench]['statuses_1'] for bench in benchmarks_keys)), axis=0)

            axes[1].scatter(x_val, y_val, alpha=0.5, s=20, color=strat_color, label=label)
            self._plot_confidence_ellipse(x_val, y_val, axes[1], strat_color)

        axes[0].set_title("ФАЗОВОЕ ПРОСТРАНСТВО ЭФФЕКТИВНОСТИ КАПИТАЛА\nКонтур А: Классический", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Максимальная историческая просадка за окно MaxDD (по модулю, %)", fontsize=11)
        axes[0].set_ylabel("Среднегодовой геометрический рост CAGR (% годовых)", fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper right')

        axes[1].set_title(f"ФАЗОВОЕ ПРОСТРАНСТВО РЕГУЛЯТОРНОГО РИСКА ЦБ\nКонтур Б: Продвинутый", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("Индекс Ульцера UI (Временной износ фонда по модулю, %)", fontsize=11)
        axes[1].set_ylabel("Чистая Альфа Дженсена (Jensen's Alpha, % годовых)", fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '03_.png'), dpi=300)
        plt.close()

        row_height = 3.2
        header_height = 1.0
        col_width_bench = 3.5
        col_width_model = 8.5

        fig_width = col_width_bench + col_width_model * len(grail_keys)
        fig_height = header_height + row_height * (len(benchmarks_keys) + 1)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_xlim(0, fig_width)
        ax.set_ylim(0, fig_height)
        ax.axis('off')

        grid_color = '#2C3E50'
        grid_width = 1.2

        y_header_top = fig_height
        y_header_bottom = fig_height - header_height

        ax.plot([0, fig_width], [y_header_top, y_header_top], color='black', linewidth=1.5)
        ax.plot([0, fig_width], [y_header_bottom, y_header_bottom], color='black', linewidth=1.5)
        ax.plot([0, 0], [y_header_bottom, y_header_top], color='black', linewidth=1.5)

        x_curr = col_width_bench
        ax.plot([x_curr, x_curr], [y_header_bottom, y_header_top], color='black', linewidth=1.5)

        for model in grail_keys:
            ax.text(x_curr + 0.4, y_header_bottom + (header_height / 2), f"Стратегия: {model}",fontsize=15, weight='bold', color=grid_color, va='center', ha='left')
            x_curr += col_width_model
            ax.plot([x_curr, x_curr], [y_header_bottom, y_header_top], color=grid_color, linewidth=grid_width)

        for r_idx, bench in enumerate(benchmarks_keys + ['1vsAll']):
            y_top = fig_height - header_height - r_idx * row_height
            y_bottom = y_top - row_height

            ax.plot([0, fig_width], [y_bottom, y_bottom], color=grid_color, linewidth=grid_width)
            ax.plot([0, 0], [y_bottom, y_top], color = grid_color, linewidth = grid_width)
            ax.text(0.3, y_bottom + (row_height / 2), bench, fontsize=13, weight='bold', color='#2C3E50', va='center', ha='left')

            x_curr = col_width_bench
            ax.plot([x_curr, x_curr], [y_bottom, y_top], color=grid_color, linewidth=grid_width)

            for model in grail_keys:
                cell_data = dict_4[model][bench]

                arr_a = cell_data['statuses_0']
                arr_b = cell_data['statuses_1']
                w_prob = cell_data['wilcoxon'] if bench != '1vsAll' else None
                n_runs = len(arr_a)

                uv_pobed_a = np.sum(arr_a == 3)
                uv_pobed_b = np.sum(arr_b == 3)

                pobed_a = np.sum(arr_a == 2)
                pobed_b = np.sum(arr_b == 2)

                nich_a = np.sum(arr_a == 1)
                nich_b = np.sum(arr_b == 1)

                porazh_a = np.sum(arr_a == 0)
                porazh_b = np.sum(arr_b == 0)

                pct_uv_a = (uv_pobed_a / n_runs) * 100
                pct_uv_b = (uv_pobed_b / n_runs) * 100

                pct_p_a = (pobed_a / n_runs) * 100
                pct_p_b = (pobed_b / n_runs) * 100

                pct_n_a = (nich_a / n_runs) * 100
                pct_n_b = (nich_b / n_runs) * 100

                pct_por_a = (porazh_a / n_runs) * 100
                pct_por_b = (porazh_b / n_runs) * 100

                y_start_text = y_top - 0.3
                y_space = 0.52
                x_pad = x_curr + 0.4

                ax.text(x_pad, y_start_text,
                        f"Значимых побед: {uv_pobed_a} — классический ({pct_uv_a:.1f}%), {uv_pobed_b} — продвинутый ({pct_uv_b:.1f}%)",
                        fontsize=11, color='#196F3D', va='top')
                ax.text(x_pad, y_start_text - y_space,
                        f"Побед: {pobed_a} — классический ({pct_p_a:.1f}%), {pobed_b} — продвинутый ({pct_p_b:.1f}%)",
                        fontsize=11, color='#27AE60', va='top')
                ax.text(x_pad, y_start_text - 2 * y_space,
                        f"Ничьих: {nich_a} — классический ({pct_n_a:.1f}%), {nich_b} — продвинутый ({pct_n_b:.1f}%)",
                        fontsize=11, color='#5D6D7E', va='top')
                ax.text(x_pad, y_start_text - 3 * y_space,
                        f"Поражений: {porazh_a} — классический ({pct_por_a:.1f}%), {porazh_b} — продвинутый ({pct_por_b:.1f}%)",
                        fontsize=11, color='#BA300D', va='top')
                if w_prob: ax.text(x_pad, y_start_text - 4 * y_space - 0.1,
                        f"Статистическая надежность превосходства: {round(w_prob, 4) if w_prob<= 99.99 else '>99.99'}%",
                        fontsize=12, weight='bold', color='#1A5276', va='top')

                x_curr += col_width_model
                ax.plot([x_curr, x_curr], [y_bottom, y_top], color=grid_color, linewidth=grid_width)

        plt.tight_layout()
        file_path = os.path.join(self.output_dir, '04_detailed_metrics_matrix.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
