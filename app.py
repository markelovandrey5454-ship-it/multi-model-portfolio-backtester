import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np
import os

SLIDES = [
    "Главная страница",
    "Траектории капитала",
    "Метрики эффективности",
    "Метрики риска",
    "Фазовые пространства",
    "Вывод исследования"
]

DATA_DIR = "data/results/strat_csv"
df_board = pd.read_csv('data/matrix/global_board_panel.csv', parse_dates=['Date']).set_index('Date')
lqdt_series = df_board['Денежный рынок(LQDT)']

STRATEGY_COLORS = {
    'Uniform 1/N': '#9E9E9E',
    'Random Monkey': '#E91E63',
    'Stochastic Momentum': '#2196F3',
    'Markowitz': '#9C27B0',
    'Base CVaR': '#FF9800',
    '0.1 prototype': '#4CAF50',
    'RankEnsemble_Frankenstein': '#FF5722'
}
target_models = ["0.1 prototype", "RankEnsemble_Frankenstein"]
benchmarks = ["Uniform 1/N", "Random Monkey", "Markowitz", "Base CVaR", "Stochastic Momentum"]

# --- ОБЩАЯ КОНФИГУРАЦИЯ (Ссылка/генерация и пр.) ---
st.set_page_config(page_title="Conference Presentation", layout="wide")

url_slide = st.query_params.get("slide", "1")
try:
    initial_slide_index = int(url_slide) - 1
    if not (0 <= initial_slide_index < len(SLIDES)): initial_slide_index = 0
except ValueError: initial_slide_index = 0

if "current_slide_index" not in st.session_state:
    st.session_state.current_slide_index = initial_slide_index

# ------- ВЕРХНЯЯ НАВИГАЦИЯ (Карточки-Слайды) -------
st.write("### 📑 Слайды презентации:")
cols = st.columns(len(SLIDES))

for idx, slide_name in enumerate(SLIDES):
    with cols[idx]:
        if st.session_state.current_slide_index == idx:
            if st.button(slide_name, key=f"btn_{idx}", type="primary", use_container_width=True):
                st.session_state.current_slide_index = idx
                st.query_params["slide"] = idx + 1
        else:
            if st.button(slide_name, key=f"btn_{idx}", type="secondary", use_container_width=True):
                st.session_state.current_slide_index = idx
                st.query_params["slide"] = idx + 1
                st.rerun()

st.markdown("---")

@st.cache_data
def load_backtest_runs():
    strategy_files = {
        '0.1 prototype': 'test_0.1 prototype.csv',
        'RankEnsemble_Frankenstein': 'test_RankEnsemble_Frankenstein.csv',
        'Base CVaR': 'test_Base CVaR.csv',
        'Markowitz': 'test_Markowitz.csv',
        'Uniform 1/N': 'test_Uniform 1_N.csv',
        'Random Monkey': 'test_Random Monkey.csv',
        'Stochastic Momentum': 'test_Stochastic Momentum.csv'
    }
    combined = []
    for label, fname in strategy_files.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Strategy'] = label
            combined.append(df)
    return pd.concat(combined, ignore_index=True)

idx_active = st.session_state.current_slide_index

# ----------------- СЛАЙД 1 (Индекс 0) -----------------
if idx_active == 0:
    st.title("📊 Стратегии оптимизации портфеля на Мосбирже")
    st.subheader("Построение надежного инвестиционного полигона")

    st.info("💡 **Пояснения к проделанной работе:**")
    st.markdown("""
    * **Тут про подход к исследованию:** А тут текст
    * **Тут про стратегии:** И тут текст
    """)

# ----------------- СЛАЙД 2 (Индекс 1) -----------------
elif idx_active == 1:
    st.title("📈 Сравнительное соревнование на 11 летнем треке")

    @st.fragment
    def render_slide_2():
        if "scen_mode" not in st.session_state:
            st.session_state.scen_mode = "Lumpsum (1 млн руб)"

        scen_options = ["Lumpsum (1 млн руб)", "DCA (Регулярные взносы)", "FIRE (Жизнь на капитал)", "Lumpsum (100 млн руб)"]
        cols_scen = st.columns(len(scen_options))

        for m_idx, m_name in enumerate(scen_options):
            with cols_scen[m_idx]:
                if st.session_state.scen_mode == m_name:
                    st.button(m_name, key=f"scen_btn_{m_idx}", type="primary", use_container_width=True)
                else:
                    if st.button(m_name, key=f"scen_btn_{m_idx}", type="secondary", use_container_width=True):
                        st.session_state.scen_mode = m_name
                        st.rerun()

        selected_scen = st.session_state.scen_mode

        initial_cap = 100_000_000.0 if "100 млн" in selected_scen else 1_000_000.0
        is_lump_mode = "Lumpsum" in selected_scen

        if "100 млн" in selected_scen:
            file_path = os.path.join(DATA_DIR, "fond.csv")
            suffix = "Lump"
        else:
            file_path = os.path.join(DATA_DIR, "human.csv")
            suffix = "Lump" if "Lumpsum" in selected_scen else ("DCA" if "DCA" in selected_scen else "FIRE")

        if os.path.exists(file_path):
            df_equity = pd.read_csv(file_path, parse_dates=['Date']).set_index('Date')
            T_days = len(df_equity)

            if is_lump_mode: fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.7, 0.3])
            else: fig = go.Figure()

            lqdt_aligned =  df_board['Денежный рынок(LQDT)']

            for col in df_equity.columns:
                if col.endswith(f"_{suffix}") or col.endswith(f"_{suffix}_tf") or col.startswith(f"Inflation_Benchmark_{suffix}"):
                    if col.startswith("Inflation_Benchmark"): base_name = "Inflation_Benchmark"
                    else: base_name = col.split(f"_{suffix}")[0]

                    color = STRATEGY_COLORS.get(base_name, '#000000')
                    type_line = 'dash' if 'Inflation' in col else 'solid'

                    final_val = df_equity[col].iloc[-1]

                    if is_lump_mode:
                        if col.endswith('_Lump') and not 'Inflation' in col:
                            daily_pct = df_equity[col].pct_change().dropna()
                            cagr_val = (final_val / initial_cap) ** (252.0 / T_days) - 1.0

                            max_dd = 0.0
                            cur_top = 1.0
                            for daily_inc in daily_pct:
                                cur_top *= 1 + daily_inc
                                if cur_top >= 1.0:
                                    cur_top = 1.0
                                elif 1.0 - cur_top > max_dd:
                                    max_dd = 1.0 - cur_top
                            calmar_val = cagr_val / max_dd if max_dd > 0 else 0.0

                            excess_r = df_equity[f"{col}_tf"].pct_change().dropna() - lqdt_aligned.loc[daily_pct.index]
                            cagr_lqdt = np.exp(np.log1p(lqdt_aligned.loc[daily_pct.index]).sum()) ** (252.0 / T_days) - 1.0
                            cagr_excess = (df_equity[f"{col}_tf"].iloc[-1] / initial_cap) ** (252.0 / T_days) - 1.0 - cagr_lqdt
                            downside_r = np.minimum(0, excess_r)
                            downside_std = np.sqrt(np.mean(downside_r ** 2))
                            if cagr_excess > 0: sortino_val = cagr_excess / (downside_std * np.sqrt(252))
                            else: sortino_val = cagr_excess * (downside_std * np.sqrt(252))

                            leg_label = f"{base_name} (CAGR: {cagr_val * 100:.1f}%, Sortino: {sortino_val:.2f}, Calmar: {calmar_val:.2f})"
                        else:
                            leg_label = f"{base_name} (CAGR: {((final_val / initial_cap) ** (252.0 / T_days) - 1.0) * 100.0:,.1f}%)"
                    else:
                        leg_label = f"{base_name} (Финальный капитал: {final_val:_.1f} руб.)"


                    if not col.endswith("_tf"):
                        trace_line = go.Scatter(
                            x=df_equity.index,
                            y=df_equity[col],
                            mode='lines',
                            name=leg_label,
                            legendgroup=col,
                            line=dict(color=color, width=2, dash=type_line),
                            hovertemplate='%{x|%Y-%m-%d}<br>Капитал: %{y:,.0f} руб<extra></extra>'
                        )

                        fig.update_layout(
                            legend=dict(
                                x=0.02,
                                y=0.98,
                                xanchor='left',
                                yanchor='top',
                                bgcolor='rgba(255, 255, 255, 0.8)',
                                bordercolor='LightGray',
                                borderwidth=1,
                                tracegroupgap=0
                            )
                        )

                        if is_lump_mode:
                            large_mask = np.abs(daily_pct) > 0.01
                            y_transformed = np.copy(daily_pct)
                            y_transformed[large_mask] = np.sign(daily_pct[large_mask]) * 0.01 * (np.log10(np.abs(daily_pct[large_mask]) / 0.01) + 1)
                            fig.add_trace(trace_line, row=1, col=1)
                            if 'Inflation' not in col:
                                fig.add_trace(go.Scattergl(
                                    x=df_equity.index[1:],
                                    y=y_transformed,
                                    customdata=daily_pct,
                                    mode='markers',
                                    name=leg_label,
                                    legendgroup=col,
                                    showlegend=False,
                                    marker=dict(color=color, size=3, opacity=0.35),
                                    hovertemplate='%{x|%Y-%m-%d}<br>Дн. доходность: %{customdata:.2%}<extra></extra>'
                                ), row=2, col=1)
                        else: fig.add_trace(trace_line)

            if is_lump_mode:
                symlog_ticks = [-0.10, -0.05, 0.05, 0.10]
                symlog_text = ["-10.0%", "-5.0%", "-1.0%", "0.0%", "1.0%", "5.0%", "10.0%"]
                transformed_ticks = np.sign(symlog_ticks) * 0.01 * (np.log10(np.abs(symlog_ticks) / 0.01) + 1)
                transformed_ticks = sorted(list(transformed_ticks) + [-0.01, 0.0, 0.01])

                fig.update_layout(
                    yaxis1=dict(title="Капитал (руб)", tickformat=",.0f", gridcolor="#E5E7EB"),
                    yaxis2=dict(
                        title="Дневная доходность",
                        type="linear",
                        tickvals=transformed_ticks,
                        ticktext=symlog_text,
                        zeroline=True,
                        zerolinecolor="black",
                        zerolinewidth=1
                    ),
                    height=780,
                    hovermode="x unified"
                )
            else:
                fig.update_layout(
                    yaxis=dict(title="Капитал (руб)", tickformat=",.0f"),
                    xaxis_title="Дата",
                    height=620,
                    hovermode="x unified"
                )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            st.markdown(f"""
                * **Тут описание стратегии {selected_scen}:** А тут текст про ее начальные данные
                """)
        else: st.warning(f"Файл `{file_path}` не найден. Проверьте папку `{DATA_DIR}`.")

    render_slide_2()

# ----------------- СЛАЙД 3 (Индекс 2) -----------------
elif idx_active == 2:
    st.title("📊 Скорость роста и Альфа моделей")

    @st.fragment
    def render_slide_3():
        if "active_metric" not in st.session_state:
            st.session_state.active_metric = "cagr"

        metrics_map = {
            'cagr': "CAGR (% годовых)",
            'sortino': "Sortino",
            'jenser': "Jensen's Alpha (% годовых)"
        }

        cols_met = st.columns(len(metrics_map))
        for m_idx, (m_key, m_label) in enumerate(metrics_map.items()):
            with cols_met[m_idx]:
                if st.session_state.active_metric == m_key:
                    st.button(m_label, key=f"metric_{m_idx}", type="primary", use_container_width=True)
                else:
                    if st.button(m_label, key=f"metric_{m_idx}", type="secondary", use_container_width=True):
                        st.session_state.active_metric = m_key
                        st.rerun()

        metric = st.session_state.active_metric
        df_runs = load_backtest_runs()

        if not df_runs.empty:
            zoom_izraelson = False
            if metric == 'sortino':
                zoom_izraelson = st.checkbox(
                    "🔬 Оптимизировать масштаб оси под модификацию Израильсона (приблизить отрицательные хвосты)",
                    value = False
                )

            fig_box = px.box(
                df_runs,
                x='Strategy',
                y=metric,
                color='Strategy',
                color_discrete_map=STRATEGY_COLORS,
                title=f"Распределение метрики: {metrics_map[metric]}"
            )

            fig_box.update_layout(
                showlegend=False,
                yaxis_title=metrics_map[metric],
                xaxis_title="Модель оптимизации"
            )

            if metric == 'sortino' and zoom_izraelson:
                fig_box.update_layout(yaxis=dict(range=[df_runs['sortino'].min() - 0.01, 0.01]))
                st.caption(f"ℹ️ *Используется верхняя граница в {0.01} для корректного отображения хвоста распределения (искаженным из-за коррективы Израильсона).*")

            st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': False})
        else: st.warning("Файлы `test_*.csv` не найдены в папке `data/results/`.")

    render_slide_3()

# ----------------- СЛАЙД 4 (Индекс 3) -----------------
elif idx_active == 3:
    st.title("🛡️ Анализ хвостовых рисков плотности распределения")

    @st.fragment
    def render_slide_4():
        if "active_risk_metric" not in st.session_state:
            st.session_state.active_risk_metric = "max_drawdown"

        risks_map = {
            'max_drawdown': "Max Drawdown (Пиковый шок)",
            'max_daily_drop': "Max Daily Drop (Хвостовой риск сессии)",
            'ulcer': "Индекс Ульцера (Временной износ капитала)"
        }
        cols_risk = st.columns(len(risks_map))

        for r_idx, (r_key, r_label) in enumerate(risks_map.items()):
            with cols_risk[r_idx]:
                if st.session_state.active_risk_metric == r_key:
                    st.button(r_label, key=f"risk_btn_{r_idx}", type="primary", use_container_width=True)
                else:
                    if st.button(r_label, key=f"risk_btn_{r_idx}", type="secondary", use_container_width=True):
                        st.session_state.active_risk_metric = r_key
                        st.rerun()

        risk_metric = st.session_state.active_risk_metric

        df_runs = load_backtest_runs()
        if not df_runs.empty:
            fig_violin = px.violin(
                df_runs,
                x='Strategy',
                y=risk_metric,
                color='Strategy',
                color_discrete_map=STRATEGY_COLORS,
                points="all",
                title=f"Ядерная оценка плотности распределения: {risks_map[risk_metric]}",
                box=False
            )

            fig_violin.update_layout(
                showlegend=True,
                yaxis_title="Значение параметра (%)",
                xaxis_title="Модель оптимизации",
                hovermode="closest"
            )

            fig_violin.update_traces(spanmode='hard')

            st.plotly_chart(fig_violin, use_container_width=True, config={'displayModeBar': False})
        else:
            st.warning("Файлы `test_*.csv` не найдены в папке `data/results/strat_csv`.")

    render_slide_4()

# ----------------- СЛАЙД 5 (Индекс 4) -----------------
elif idx_active == 4:
    st.title("🧬 Фазовые пространства эффективности и риска")

    @st.fragment
    def render_slide_5():
        if "active_contour_mode" not in st.session_state:
            st.session_state.active_contour_mode = "contour_a"

        contours_map = {
            'contour_a': "Контур А: Классический (CAGR vs Max Drawdown)",
            'contour_b': "Контур Б: Продвинутый / Риск ЦБ (Jensen's Alpha vs Ulcer Index)"
        }
        cols_cont = st.columns(len(contours_map))

        for c_idx, (c_key, c_label) in enumerate(contours_map.items()):
            with cols_cont[c_idx]:
                if st.session_state.active_contour_mode == c_key:
                    st.button(c_label, key=f"cont_btn_{c_idx}", type="primary", use_container_width=True)
                else:
                    if st.button(c_label, key=f"cont_btn_{c_idx}", type="secondary", use_container_width=True):
                        st.session_state.active_contour_mode = c_key
                        st.rerun()

        contour_mode = st.session_state.active_contour_mode
        df_runs = load_backtest_runs()

        if not df_runs.empty:
            fig_scatter = go.Figure()

            for strat_name, color in STRATEGY_COLORS.items():
                strat_df = df_runs[df_runs['Strategy'] == strat_name]

                if contour_mode == 'contour_a':
                    x_vals = np.abs(strat_df['max_drawdown'].to_numpy())
                    y_vals = strat_df['cagr_taxless'].to_numpy() - strat_df['cagr_lqdt'].to_numpy()

                    x_title, y_title = "Максимальная историческая просадка MaxDD (%)", "Среднегодовой геометрический рост CAGR (%)"
                    title_text = "Фазовое пространство эффективности капитала (Контур А)"
                else:
                    x_vals = np.abs(strat_df['ulcer'].to_numpy())
                    y_vals = strat_df['jenser'].to_numpy()

                    x_title, y_title = "Индекс Ульцера UI (Временной износ фонда, %)", "Чистая Альфа Дженсена (Jensen's Alpha, % годовых)"
                    title_text = "Фазовое пространство регуляторного риска ЦБ (Контур Б)"

                fig_scatter.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='markers',
                    name=strat_name,
                    legendgroup=strat_name,
                    showlegend=False,
                    marker=dict(color=color, size=5, opacity=0.4),
                    hovertemplate=f"<b>{strat_name}</b><br>{x_title}: %{{x:.2f}}%<br>{y_title}: %{{y:.2f}}%<extra></extra>"
                ))

                median_x = np.median(x_vals)
                median_y = np.median(y_vals)

                fig_scatter.add_trace(go.Scatter(
                    x=[median_x], y=[median_y],
                    mode='markers',
                    name=f"{strat_name}",
                    legendgroup=strat_name,
                    showlegend=True,
                    marker=dict(color=color, size=12, symbol='square', line=dict(color='black', width=1)),
                    hovertemplate=f"<b>Центр масс: {strat_name}</b><br>Медиана X: {median_x:.2f}%<br>Медиана Y: {median_y:.2f}%<extra></extra>"
                ))

                cov = np.cov(x_vals, y_vals)
                mean_x = np.mean(x_vals)
                mean_y = np.mean(y_vals)

                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                order = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[order]
                eigenvectors = eigenvectors[:, order]

                angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

                n_std = 2.45
                rad_x = n_std * np.sqrt(np.maximum(eigenvalues[0], 0))
                rad_y = n_std * np.sqrt(np.maximum(eigenvalues[1], 0))

                theta = np.linspace(0, 360, 100)
                ellipse_x = rad_x * np.cos(np.radians(theta))
                ellipse_y = rad_y * np.sin(np.radians(theta))

                cos_a = np.cos(np.radians(angle))
                sin_a = np.sin(np.radians(angle))
                R = np.array([[cos_a, -sin_a],
                              [sin_a, cos_a]])

                raw_points = np.column_stack([ellipse_x, ellipse_y])
                ellipse_points = np.dot(raw_points, R.T)
                ellipse_points[:, 0] += mean_x
                ellipse_points[:, 1] += mean_y

                fig_scatter.add_trace(go.Scatter(
                    x=ellipse_points[:, 0], y=ellipse_points[:, 1],
                    mode='lines',
                    name=strat_name,
                    line=dict(color=color, width=1.5, dash='dash'),
                    legendgroup=strat_name,
                    showlegend=False,
                    visible=True,
                    hoverinfo='skip'
                ))

            fig_scatter.update_layout(
                title=title_text,
                xaxis_title=x_title,
                yaxis_title=y_title,
                hovermode="closest"
            )

            st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})
            st.caption("ℹ️ *Пунктирные линии отображают 95% границы устойчивости (доверительные эллипсы распределения).*")
        else:
            st.warning("Файлы `test_*.csv` не найдены в папке `data/results/`.")

    render_slide_5()

# ----------------- СЛАЙД 6 (Индекс 5) -----------------
elif idx_active == 5:
    st.title("🎯 Финальное сравнение: как часто и насколько решительно")


    @st.fragment
    def render_slide_6():
        df = load_backtest_runs()
        key = ["start_date", "end_date", "encoded_assets"]

        c1, c2, c3, c4 = st.columns(4)
        contour = c1.radio("Аналитический контур", ["A", "B"], horizontal=True,
                           format_func=lambda c: "А: CAGR и MaxDD" if c == "A" else "Б: Jensen и Ulcer")
        p = c2.select_slider("Степень жесткости риска (p)", options=[1.0, 1.5, 2.0], value=2.0)
        low = c3.slider("Порог «победы» (перцентиль шума)", 75, 95, 80)
        regime = c4.radio("Окна", ["Все", "С 24.03.2022", "Без 24.03.2022"])
        model = st.radio("Модель", target_models, horizontal=True)

        has_shock = (df["start_date"] < "2022-03-24") & (df["end_date"] >= "2022-03-24")
        if regime == "С 24.03.2022":
            df = df[has_shock]
        elif regime == "Без 24.03.2022":
            df = df[~has_shock]

        cagr = df.pivot_table(index=key, columns="Strategy", values="cagr_taxless", aggfunc="first")
        jen = df.pivot_table(index=key, columns="Strategy", values="jenser", aggfunc="first")
        dd = df.pivot_table(index=key, columns="Strategy", values="max_drawdown", aggfunc="first").abs()
        ui = df.pivot_table(index=key, columns="Strategy", values="ulcer", aggfunc="first").abs()
        lq = df.groupby(key)["cagr_lqdt"].first()

        bench = [b for b in benchmarks if b in cagr.columns]
        if contour == "A":
            ret, risk = cagr, dd
            K = cagr.sub(lq, axis=0) / dd ** p
            ret_name, risk_name = "CAGR, % годовых", "MaxDD, п.п."
        else:
            ret, risk = jen, ui
            K = jen / ui ** p
            ret_name, risk_name = "Альфа Дженсена, % годовых", "Ulcer, п.п."
        K = K.dropna(subset=[model] + bench)
        st.caption(f"Окон в выборке: {len(K)}.")

        placebo = (K["Uniform 1/N"] - K["Random Monkey"]).to_numpy()
        noise = np.concatenate([placebo, -placebo])
        lo, hi = np.percentile(noise, [low, 99.9])

        rows, codes = {}, []
        pairs = {"Эталон шума": noise}
        for b in bench: pairs[b] = (K[model] - K[b]).to_numpy()
        for name, d in pairs.items():
            code = np.select([d >= hi, d >= lo, d > -lo, d > -hi], [4, 3, 2, 1], default=0)
            rows[name] = [100 * np.mean(code == i) for i in range(5)]
            if name in bench: codes.append(code)
        worst = np.min(codes, axis=0)
        rows["1vsAll: худшая среди бенчмарков"] = [100 * np.mean(worst == i) for i in range(5)]

        st.subheader("Как часто и насколько решительно модель обгоняет бенчмарк")
        fig = go.Figure()
        start = {n: -(r[0] + r[1] + r[2] / 2) for n, r in rows.items()}
        for i, (cat, color) in enumerate(zip(["Уверенное поражение", "Поражение", "Ничья", "Победа", "Уверенная победа"],
                                             ["#B71C1C", "#EF9A9A", "#E0E0E0", "#A5D6A7", "#1B5E20"])):
            fig.add_trace(go.Bar(
                y=list(rows),
                x=[r[i] for r in rows.values()],
                base=[start[n] for n in rows],
                orientation="h",
                name=cat,
                marker_color=color,
                hovertemplate="<b>%{y}</b><br>%{data.name}: <b>%{customdata:.2f}%</b><extra></extra>",
                customdata=[r[i] for r in rows.values()]
            ))
            for n, r in rows.items(): start[n] += r[i]
        fig.update_layout(
            barmode="overlay",
            height=130 + 62 * len(rows),
            xaxis_title="% окон",
            yaxis=dict(autorange="reversed"),
            legend=dict(orientation="h", x=0.15, y=-0.15)
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        #st.dataframe(pd.DataFrame(rows, index=CATS).T.round(1), use_container_width=True)
        st.caption(f"«Победа/Поражение» — разность k не меньше {low}-го перцентиля шума ({lo:.4f}), «Уверенная победа/поражение» — не меньше 99.9-го ({hi:.4f}).  \n"
            "Шум — разность Uniform 1/N и Random Monkey на тех же окнах (верхняя строка показывает, распределение разности этих стратегий)")

        st.subheader("Цена риска: при каком отношении к риску вердикт меняется")
        grid = np.arange(1, 20.01, 0.05)
        lam_rows = {}
        for b in bench:
            dc = (ret[model] - ret[b]).reindex(K.index).to_numpy()
            dr = (risk[model] - risk[b]).reindex(K.index).to_numpy()
            wins = ((dc[None, :] - grid[:, None] * dr[None, :]) > 0).mean(axis=1)
            above = wins > 0.5
            if above.all():
                verdict = "Лучше при любой цене риска"
            elif not above.any():
                verdict = "Чуже при любой цене риска"
            else: verdict = f"λ* ≈ {grid[np.argmax(above[1:] != above[:-1]) + 1]:.2f}"
            lam_rows[b] = {"Вердикт": verdict,
                           "Гарантированная доля, %": round(100 * wins.min(), 1)}
        st.dataframe(pd.DataFrame(lam_rows).T, use_container_width=True)
        st.caption(f"λ — сколько п.п. «{ret_name}» инвестор готов отдать за 1 п.п. «{risk_name}». Гарантированная доля — минимум доли побед по всем λ.")

    render_slide_6()

# ------- НИЖНЯЯ НАВИГАЦИЯ (Переключение слайдов) -------
st.markdown("---")
col_prev, _, col_next = st.columns([1, 4, 1])

prev_clicked = False
next_clicked = False

if st.session_state.current_slide_index > 0:
    if col_prev.button("⬅️ Назад", key="bottom_prev", use_container_width=True):
        prev_clicked = True

if st.session_state.current_slide_index < len(SLIDES) - 1:
    if col_next.button("Вперед ➡️", key="bottom_next", use_container_width=True):
        next_clicked = True

st.components.v1.html(
    """
    <script>
    const doc = window.parent.document;

    function keyHandler(e) {
        if (e.key === "ArrowLeft") {
            const buttons = doc.querySelectorAll("button");
            for (let btn of buttons) {
                if (btn.innerText.includes("Назад")) {
                    btn.click();
                    break;
                }
            }
        } else if (e.key === "ArrowRight") {
            const buttons = doc.querySelectorAll("button");
            for (let btn of buttons) {
                if (btn.innerText.includes("Вперед")) {
                    btn.click();
                    break;
                }
            }
        }
    }

    doc.removeEventListener("keydown", keyHandler);
    doc.addEventListener("keydown", keyHandler);
    </script>
    """,
    height=0,
)

if prev_clicked:
    st.session_state.current_slide_index -= 1
    st.query_params["slide"] = st.session_state.current_slide_index
    st.rerun()

if next_clicked:
    st.session_state.current_slide_index += 1
    st.query_params["slide"] = st.session_state.current_slide_index
    st.rerun()
