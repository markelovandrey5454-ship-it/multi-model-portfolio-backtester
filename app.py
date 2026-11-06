import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_javascript import st_javascript
from user_agents import parse

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

DATA_DIR = "data/results"

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
except ValueError:
    initial_slide_index = 0

if "current_slide_index" not in st.session_state:
    st.session_state.current_slide_index = initial_slide_index
if "expand_all_state" not in st.session_state:
    st.session_state.expand_all_state = False

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

# ----- ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ (Для оптимизации) -----
@st.cache_data
def load_backtest_runs():
    strategy_files = {
        '0.1 prototype': 'test_0.1 prototype.parquet',
        'RankEnsemble_Frankenstein': 'test_RankEnsemble_Frankenstein.parquet',
        'Base CVaR': 'test_Base CVaR.parquet',
        'Markowitz': 'test_Markowitz.parquet',
        'Uniform 1/N': 'test_Uniform 1_N.parquet',
        'Random Monkey': 'test_Random Monkey.parquet',
        'Stochastic Momentum': 'test_Stochastic Momentum.parquet'
    }
    combined = []
    for label, fname in strategy_files.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df['Strategy'] = label
            combined.append(df)
    return pd.concat(combined, ignore_index=True)


@st.cache_data
def load_lqdt():
    df_board = pd.read_parquet('data/global_board_panel.parquet').set_index('Date')
    return df_board['Денежный рынок(LQDT)']


@st.cache_data
def load_and_calculate_equity(file_path, suffix, initial_cap, is_lump_mode):
    df_equity = pd.read_parquet(file_path).set_index('Date')
    T_days = len(df_equity)

    lqdt_aligned = load_lqdt()

    df_pct_all = df_equity.pct_change()
    lqdt_sub = lqdt_aligned.reindex(df_pct_all.index)
    cagr_lqdt = np.exp(np.log1p(lqdt_sub).sum()) ** (252.0 / T_days) - 1.0

    running_maxes = df_equity.cummax()
    drawdowns_all = 1.0 - df_equity / running_maxes
    max_dds = drawdowns_all.max()

    labels_dict = {}

    for col in df_equity.columns:
        if col.endswith(f"_{suffix}") or col.endswith(f"_{suffix}_tf") or col.startswith(f"Inflation_Benchmark_{suffix}"):
            if col.startswith("Inflation_Benchmark"): base_name = "Inflation_Benchmark"
            else: base_name = col.split(f"_{suffix}")[0]

            final_val = df_equity[col].iloc[-1]

            if is_lump_mode:
                if col.endswith('_Lump') and not 'Inflation' in col:
                    cagr_val = (final_val / initial_cap) ** (252.0 / T_days) - 1.0
                    max_dd = max_dds[col]
                    calmar_val = cagr_val / max_dd if max_dd > 0 else 0.0

                    excess_r = df_pct_all[f"{col}_tf"].dropna() - lqdt_sub
                    cagr_excess = (df_equity[f"{col}_tf"].iloc[-1] / initial_cap) ** (252.0 / T_days) - 1.0 - cagr_lqdt
                    downside_std = np.sqrt(np.mean(np.minimum(0, excess_r) ** 2))

                    sortino_val = cagr_excess / (downside_std * np.sqrt(252)) if cagr_excess > 0 else cagr_excess * (downside_std * np.sqrt(252))

                    if abs(sortino_val) < 0.01: sortino_str = f"{sortino_val:.5f}"
                    else: sortino_str = f"{sortino_val:.2f}"
                    
                    labels_dict[col] = f"{base_name} (CAGR: {cagr_val * 100:.1f}%, Sortino: {sortino_str}, Calmar: {calmar_val:.2f})"
                else: labels_dict[col] = f"{base_name} (CAGR: {((final_val / initial_cap) ** (252.0 / T_days) - 1.0) * 100.0:,.1f}%)"
            else: labels_dict[col] = f"{base_name} (Финальный капитал: {final_val:_.1f} руб.)"

    return df_equity, labels_dict, df_pct_all

ua_string = st_javascript("window.navigator.userAgent;")

if ua_string: mobile = parse(ua_string).is_mobile
else: mobile = False

st.html(
"""
<style>
.stExpander details summary p {
    font-size: 16px !important;
    line-height: 1.5 !important;
}
</style>
""")
if mobile: 
    st.html("""
    <style>
    .stAppHeader h1, h1, [data-testid="stHeader"] h1 {
        font-size: 30px !important;
    }
    </style>
    """)


idx_active = st.session_state.current_slide_index

# ----------------- СЛАЙД 1 (Индекс 0) -----------------
if idx_active == 0:
    st.title("Методология моделирования торговой среды и спецификация оцениваемых стратегий")


    @st.fragment
    def render_slide_1():
        if "info_mode" not in st.session_state:
            st.session_state.info_mode = "Контур очистки и нормализации данных"

        st.info("💡 **Пояснения к проделанной работе**")
        info_options = ["Контур очистки и нормализации данных", "Инфраструктурный контур бэктестирования", "Архитектура оцениваемых моделей"]
        cols_info = st.columns(len(info_options))

        for m_idx, m_name in enumerate(info_options):
            with cols_info[m_idx]:
                if st.session_state.info_mode == m_name:
                    st.button(m_name, key=f"scen_btn_{m_idx}", type="primary", use_container_width=True)
                else:
                    if st.button(m_name, key=f"scen_btn_{m_idx}", type="secondary", use_container_width=True):
                        st.session_state.info_mode = m_name
                        st.rerun()
        if st.session_state.info_mode == "Контур очистки и нормализации данных":
            with st.expander("**Обеспечение репрезентативности выборки и устранение селективного искажения**  "
                "\n  Формирование кросс-секционной исторической панели из 90+ макроинструментов, включающей полную сквозную историю с динамическим учетом листинга, делистинга и банкротств.",
                expanded=st.session_state.expand_all_state):
                st.write("""
                Вместо стандартного использования фиксированного списка текущих компонентов индекса Мосбиржи, смоделирована динамическая вселенная активов за период 2010–2026 гг.
                * **Диверсификация по классам активов:** Панель включает акции (от SBER и до BAZA), долговые бенчмарки (реализованные через индексы цен с TR: RGBITR, RUGBITR10Y, RUCBITRB/RUCBHYTR), 
                    товарный рынок (GLDRUB, SLVRUB), валютную секцию (CNYRUB, USDRUB, EURRUB) и денежный рынок (с единым игроком в виде модифицированного LQDT).
                * **Учет терминальных состояний эмитентов:** Реализовано строгое математическое описание деструкции капитала на примере реальных банкротств («Трансаэро», «Обувь России») 
                    и делистингов («Полиметалла», «Соликамского магниевого завода») рынка РФ. Принудительный/добровольный выкуп при этом моделировался с учетом лагов и ликвидационной стоимости с удержанием налога.
                * **Сшивка исторических тикеров:** Алгоритм осуществляет непрерывный переход прав и ценовых рядов при смене режимов торгов (SBER-2007 → SBER03 → SBER) 
                    и корпоративных трансформациях (TCSG → T, YNDX → YDEX, FIVE → X5; MAIL → VKCO, SYNG → BELU).
                """)

            with st.expander("**Математическое обогащение данных и генерация пропущенных метрик**  "
                "\n  Восстановление непрерывной истории безрискового кэш-бенчмарка через ретроспективное дисконтирование по ставкам РЕПО и расчет робастной внутридневной волатильности по методу Паркинсона с учетом цен закрытия.",
                expanded=st.session_state.expand_all_state):
                st.write("""
                * **Синтетический кэш-бенчмарк (LQDT + MOEXREPO):** Для БПИФ денежного рынка воссоздана историческая траектория цен до даты его фактического листинга (2022 г.). 
                    Расчет реализован через ретроспективное дисконтирование базовой цены IPO по ежедневным ставкам междилерского РЕПО с ЦК (MOEXREPO) по формуле сложного процента:
                    """)
                st.latex(r"P_{t}=\frac{P_{t+1}}{\left(1 + \frac{\text{Rate}_{t}}{100}\right)^\frac{1}{252}}")
                st.write("""
                * **Микроструктурный риск (Волатильность Паркинсона):** Вместо классического оценщика Close-to-Close, занижающего риск внутри дня, внедрен робастный экстремальный метод на основе цен High и Low
                    учитывающий также цены CLOSE прошлого дня, общая формула:
                """)
                st.latex(r"\sigma_t = \sqrt{\frac{1}{4\ln 2} \cdot \left(\ln \frac{\max(High_t, Close_{t-1})}{\min(Low_t, Close_{t-1})}\right)^2}")

            with st.expander("**Инфраструктурное приближение к реальному рынку РФ**  "
                "\n  Сквозное моделирование с учетом изменений в торках Мосбиржи через динамические клиринговые лаги (T+N), фильтрацию корпоративных действий (сплитов/консолидаций), изоляцию рыночных форс-мажоров и мультивалютный кросс-клиринг дивидендов.",
                expanded=st.session_state.expand_all_state):
                st.write("""
                Инфраструктурный контур симулирует физические ограничения и правила российского биржевого рынка:
                * **Алгоритмическая нормализация сплитов:** Эвристический цикл сканирует ценовую панель на аномальные скачки (в диапазоне <= 0.4 или >= 2.5). 
                При обнаружении разрыва алгоритм подбирает коэффициент сплита и ретроспективно корректирует всю историческую цепочку цен назад, исключая ложные скачки цен.
                """)
                st.write("""
                * **Динамический клиринг (T+N) и мультивалютные дивиденды:** Учтена эволюция расчетных лагов Мосбиржи. Дивидендный поток, очищенный от НДФЛ, рассчитывается по формуле:
                """)
                st.latex(r"\Delta t_{\text{clearing}}=\begin{cases}0,&t<\text{30.08.2013 }\\ -2,&\text{30.08.2013}\le t<\text{01.08.2023 }\\ -1,&t\ge \text{01.08.2023 }\end{cases}")
                st.write("""
                Что соответствует режимам Т+0, Т+2 и Т+1 соответственно. Также для компаний выплачивавших дивиденды в валюте, реализован автоматический поиск курса на точную дату выплаты с последующей конвертацией в рубли перед реинвестированием.
                * **Изоляция рыночных форс-мажоров:** Из матриц полностью удалены дни принудительной приостановки торгов на Мосбирже (в основном февраль-март 2022 года). Это предотвращает математические искажения (смещение средних, «залипание» волатильности) в данных.

                """)

            with st.expander("⚠️ **Методологические ограничения исследования и точки роста**  "
                "\n  Фиксация фискальных параметров (НДФЛ 13%), консервативное моделирование налоговых режимов без учета льгот (ЛДВ, ИИС) и допущение об абсолютной ликвидности макроинструментов.",
                expanded=st.session_state.expand_all_state):
                st.write("""
                Честное описание допущений и зон развития математической модели, подтверждающее методологическую зрелость бэктеста:
                * **Дискретизация фискальных условий:** В бэктестере заложена фиксированная ставка всех видов налогов в размере 13%. Модель не учитывает введение в РФ прогрессивной шкалы налогообложения, а также налоги для нерезидентов РФ - 
                что может незначительно завысить конечный чистый капитал стратегий.
                * **Специфика институционального и частного контуров:** В симуляции отключены льготы долгосрочного владения (ЛДВ), налоговые вычеты по ИИС и сальдирование убытков. Это означает, что розничный сценарий рассчитывается по консервативной нижней границе. 
                При этом для банковского или фондового управления (ПИФ/ДУ) внутрифондовый налог на прибыль и дивиденды по закону равен 0%, поэтому реальные результаты институционалов будут выше.
                * **Абсолютная ликвидность макроинструментов:** Принято допущение о безграничной ликвидности для квази-кэша (LQDT), валют, металлов и облигаций (представленных в виде индексов). Хотя это и было сделано осознано, 
                ведь в реальной практике крупный фонд не покупает сами индексы, а собирает портфель из наиболее ликвидных выпусков ОФЗ/ВДО аналогичной дюрации или валют/металлов/кэша - что имеют динамику близкую к индексам на достаточно больших временных окнах.
                """)

        elif st.session_state.info_mode == "Инфраструктурный контур бэктестирования":
            with st.expander("**Избегаем критических ошибок моделирования**  "
                "\n  Исключение подглядываний в будущее, итерационный расчет транзакционного трения через систему нелинейных уравнений и ежегодный налоговый клиринг капитала.",
                expanded=st.session_state.expand_all_state):
                st.write("""
                Архитектурные решения бэктестера, гарантирующие математическую строгость и причинно-следственную чистоту процесса симуляции:
                * **Строгая временная изоляция:** Информационный контур моделей жестко разделен. Генерация целевых весов и оптимизация портфеля происходят в конце сессии t-1 на исторических данных, как их фактическое исполнение, 
                начисление доходности (к ценам в конце сессии t) и изменения весов. Попадание данных «из будущего» в оптимизаторы исключено.
                * **Самосогласованный расчет транзакционных издержек:** Вместо линейного вычитания фиксированной комиссии брокера, алгоритм решает систему нелинейных уравнений с поиском фиксированной точки. 
                Это позволяет определить точный итоговый капитал с учетом знака и объема изменения позиций по каждому активу, предотвращая «виртуальное» завышение доходности и помогает привести к новым весам в точности продиктовынным моделями:
                """)
                st.latex(r"S_{t}=S_{t-1}\cdot \frac{1+c\cdot \sum _{i=1}^{N} \mathrm{sgn}(\Delta x_{i})\cdot w_{i,t-1}}{1+c\cdot \sum_{i=1}^{N} \mathrm{sgn}(\Delta x_{i})\cdot w_{i,t}}")
                st.write("""
                * **Фискальный клиринг:** Реализовано дискретное списание налога на прирост капитала строго на границах календарных финансовых лет с автоматическим пересчетом чистой налоговой базы портфеля.
                """)

            with st.expander("**Институциональный уровень и ограничения ликвидности**  "
                "\n  Принудительное приведение моделей к структурным нормативам Банка России, динамический контроль емкости стратегий по объемам торгов стакана и моделирование хронологического лага выплат.",
                expanded=st.session_state.expand_all_state):
                st.write("""
                Вывод бэктестера на уровень промышленного управления капиталом (институциональный фонд/банк) за счет внедрения физических и нормативных ограничений:
                * **Регуляторные лимиты ЦБ РФ:** В контур оптимизации встроены жесткие требования Банка России для паевых инвестиционных фондов. Совокупный вес валютных инструментов ограничен уровнем <= 20%, 
                доля драгоценных металлов <= 15%, а обязательное удержание высоколиквидного кэша зафиксировано на уровне >= 5%.
                * **Динамический ликвидный барьер капитала:** Защита от рыночного проскальзывания при масштабировании стратегий реализована через расчет предельной емкости позиций. 
                Максимальный вес каждого актива динамически ограничивается под текущий объем портфеля на основе 5-дневного и 22-дневного скользящих оборотов торгов:
                """)
                st.latex(r"w_{i,t}(max)=\frac{0.015\cdot \left(0.8\cdot \min (\text{Vol}_{5},\text{Vol}_{22})+0.2\cdot \max (\text{Vol}_{5},\text{Vol}_{22})\right)}{S_t}")
                st.write("""
                * **Хронологический лаг выплат и терминальное дисконтирование:** Начисленные дивиденды и ликвидационная стоимость акций при делистинге блокируются в симуляционной очереди, для дивидендов это 15 торговых дней. 
                Если дата выплаты выходит за финальную границу бэктеста, кэш математически дисконтируется обратно к терминальной дате по сложной ставке денежного рынка.
                * **Ограничение скорости исполнения:** Скорость перетока капитала между ребалансировками жестко лимитирована по объему (<= 10% в обычные дни или <= 30% в дни повышенной волатильности от дневного объема торгов актива), 
                что не позволяет модели мгновенно развернуть крупную позицию в неликвидном инструменте.
                """)

            with st.expander("⚠️ **Методологические ограничения исследования и точки роста**  "
                "\n  Использование постоянной базовой ставки комиссий брокера, отсутствие моделирования микроструктуры стакана и консервативное Long-Only моделирование без маржинальных плеч.",
                expanded=st.session_state.expand_all_state):
                st.write("""
                Границы применимости разработанной инфраструктурной среды и направления для будущей модернизации кода:
                * **Линейный масштаб издержек:** Издержки зафиксированы на уровне 0.05% за сделку. В реальной практике институциональных фондов тарифная сетка брокеров и биржи является регрессивной, 
                что делает текущие расчеты транзакционного трения консервативными и смещенными в пользу избыточных расходов.
                * **Идеализированная микроструктура стакана (отсутствие спреда):** Модель бэктестирования производит расчеты по ценам закрытия и не учитывает разницы внутридневных цен покупки/продажи. 
                В условиях экстремальных хвостовых рисков или низкой ликвидности расширение спреда может приводить к завышениею капитала (недооценке потерь), что является зоной развития бэктестера.
                * **Исключение маржинального кредитования:** Симуляция накладывает жесткое ограничение на неотрицательность весов. Отсутствие механизмов шорта и маржинальных плеч исключает риски принудительного закрытия позиций по Margin Call, 
                но ограничивает способность моделей монетизировать растущие/падающие тренды рынка РФ.
                """)

        elif st.session_state.info_mode == "Архитектура оцениваемых моделей":
            with st.expander("**Наивно диверсифицированный портфель (Uniform 1/N)**  "
                "\n  Классический эталон пассивного инвестирования с дискретным эквивесовым распределением капитала по доступной кросс-секционной панели активов.",
                expanded=st.session_state.expand_all_state):
                st.write("""
                * **Динамическое квантование весов:** На каждом шаге ребалансировки алгоритм определяет точное количество торгуемых в данный день эмитентов и жестко фиксирует долю каждого инструмента:
                """)
                st.latex(r"w_{i,t}=\frac{1}{N_{t}}")
                st.write("""
                * **Автоматическая адаптация вселенной активов:** Модель не требует ручной настройки при IPO или делистингах. При появлении новых тикеров или терминальном уходе компаний кол-во доступных активов мгновенно пересчитывается, а капитал перераспределяется.
                """)

            with st.expander("**Симуляция случайного блуждания (Random Monkey)**  "
                "\n  Стохастический стресс-бенчмарк с пониженной частотой инвестирования и генерацией абсолютно случайных весов портфеля.",
                expanded=st.session_state.expand_all_state):
                st.write("""
                * **Стохастическое квантование весов:** Раз в месяц алгоритм генерирует случайный вектор долей для всех доступных на Мосбирже активов, который затем принудительно нормализуется к единице:
                """)
                st.latex(r"\sum _{i=1}^{N}w_{i,t}=1.0")
                st.write("""
                * **Пониженная частота изменения весов:** Стратегия намеренно ребалансируется дискретно (один раз в месяц в отличие от ежедневного пересчета остальных моделей). Это сделано для того, чтобы поддержать обычную случайную стратегию 
                (что теряет капитал при ребалансировках подобных прочим), а также сопоставить результаты с более активными стратегиями.
                """)

            with st.expander("**Стратегия стохастического моментума (Stochastic Momentum)**  "
                "\n  Классический трендовый бенчмарк, перераспределяющий капитал на основе ежедневной оценки силы и импульса ценовых трендов активов.",
                expanded=st.session_state.expand_all_state):
                st.write("""
                * **Импульсное квантование весов:** Алгоритм ежедневно рассчитывает моментум (скорость изменения цен) по каждому эмитенту за фиксированное скользящее окно. Капитал распределяется среди активов, демонстрирующих устойчивый восходящий тренд.
                * **Терминальный защитный триггер:** Ключевая особенность алгоритма — логика полной капитуляции при системных шоках. Если в скользящем окне ни один рисковый актив не показывает положительного математического ожидания, 
                модель мгновенно обнуляет доли всех акций и переводит капитал в инструмент денежного рынка (LQDT), защищая капитал от просадок.
                """)

            with st.expander("**Кастомизированный оптимизатор Марковица (Markowitz)**  "
                "\n  Классическая квадратичная оптимизация баланса риска и доходности, модернизированная за счет адаптивного таргетирования, робастного сжатия ковариаций Ледуа-Вулфа и прямого штрафа за транзакционный оборот портфеля.",
                expanded=st.session_state.expand_all_state):
                st.markdown("<div id='target_l1'></div>", unsafe_allow_html=True)
                st.write("""
                * **Стратегическое квантование весов:** Стратегия базируется на классической современной портфельной теории, где инвестирование — это задача нахождения набора активов, 
                который минимизирует общую дисперсию портфеля при заданном целевом уровне ожидаемой доходности. Достижение таргета контролируется адаптивно: если на исследуемой области в 252 торговых дня жесткая планка дневной доходности становится недостижимой 
                в периоды обвала рынка РФ, алгоритм динамически снижает планку до рыночной медианы.
                """)
                st.write("""* **L1-штраф за оборот:** В отличие от стандартных моделей, которые могут полностью перетряхивать портфель ради микроскопического улучшения математического ожидания, в целевую функцию встроен прямой штраф за избыточную торговлю. 
                Оптимизатор сопоставляет выгоду от изменения весов с реальной брокерской комиссией, подавляя паразитный оборот и удерживая веса в стабильном состоянии:
                """)
                st.latex(r"min\left(c\cdot \sum _{i=1}^{N}|w_{i}-prev\_weights_{i}|\right)")
                st.write("""
                * **Робастное сжатие Ледуа-Вулфа:** Вместо использования сырой исторической ковариационной матрицы, которая крайне зашумлена и склонна порождать экстремальные ложные веса, алгоритм применяет математическое сжатие к целевой матрице. 
                Это стабилизирует квадратичную форму оптимизатора при расчете портфельного риска и обеспечивает устойчивость весов на выборках с высокой волатильностью.
                """)

            with st.expander("**Линейный оптимизатор условной стоимости под риском (Base CVaR)**  "
                "\n  Оптимизация структуры портфеля на основе минимизации ожидаемых хвостовых убытков (β = 0.95), модернизированная за счет жесткого ограничения концентрации акций и принудительного удержания баланса рисковых активов.",
                expanded=st.session_state.expand_all_state):
                st.write("""
                * **Стратегическое квантование весов:** Подход CVaR направлен исключительно на минимизацию левого хвоста распределения доходностей. Модель рассчитывает ожидаемый убыток в 5% самых худших торговых сессий за скользящее окно в 252 торговых дня, 
                линеаризуя задачу оптимизации через вспомогательные переменные убытков и квантильного порога:
                """)
                st.latex(r"\text{CVaR}_{\beta,i}=\alpha +\frac{1}{1-\beta }\cdot \frac{1}{T}\sum _{t=1}^{T}u_{t}")
                st.write("""
                * **Преодоление патологической консервативности:** Стандартный линейный CVaR на рынке РФ склонен полностью капитулировать перед хвостовыми рисками, уходя на 100% в денежный рынок. 
                Для преодоления этой проблемы в целевую функцию минимизации встроен регуляризатор-штраф, подтягивающий долю LQDT в район 40%:
                """)
                st.latex(r"\text{min}(|w_{\text{LQDT}}-0.4|)")
                st.markdown("""
                * **L1-штраф за оборот:** Работает также как и в прочих институциональных стратегиях (<a href="#target_l1" target="_self">см. описание выше</a>).
                """, unsafe_allow_html=True)

            with st.expander("**Прогностический ранговый ML-ансамбль (RankEnsemble_Frankenstein)**  "
                "\n  Активная прогностическая стратегия на базе ансамбля линейных моделей (Ridge) и градиентного бустинга (LightGBM), генерирующая избыточную доходность через ежедневное ранговое предсказание ценового импульса.",
                expanded=st.session_state.expand_all_state):
                st.write("""
                * **Стратегическое квантование весов:** Алгоритм объединяет линейные паттерны Ridge-регрессии и нелинейные зависимости древесного бустинга LightGBM. Чтобы избежать проблемы масштабирования сырых предсказаний, 
                модель переходит в пространство рангов, а веса моделей динамически адаптируются через функцию Softmax на основе скользящего коэффициента информационной эффективности за окно в 63 торговые сессии (3 месяца):
                """)
                st.latex(r"w_{m}=\frac{\exp (20\cdot {IC}_{m})}{\sum _{k\in m}\exp (20\cdot {IC}_{k})},\quad m\in \{\text{Ridge},\text{LGBM}\}")
                st.write("""
                * **Компонентный базис ансамбля (архитектура Ridge и LightGBM):** \n  1) Параметрический блок (Ridge) обучается на скользящем окне в пространстве Z-оценок, предсказывая не доходность актива, а относительную силу каждого актива внутри дня. 
                Для подавления рыночного шума мгновенные коэффициенты текущего шага не используются — вместо этого алгоритм производит историческую свертку весов β с экспоненциально затухающим фактором забывания (0.95), что стабилизирует вектор предсказаний при ценовых импульсах:
                """)
                st.latex(r"\beta _{\text{smooth}}=\sum _{t=1}^{T}\beta _{t}\cdot \frac{0.95^{T-t}}{\sum _{\tau=1}^{T}0.95^{T-\tau }}")
                st.write("""
                2) Нелинейный блок (LightGBM) разворачивает матрицы признаков всех активов в единый сквозной вектор (панель). Это позволяет выявлять универсальные нелинейные зависимости для всего рынка одновременно, кратно увеличивая выборку данных. 
                В вектор признаков LightGBM принудительно внедряется общая скользящая доходность всего рынка за 5 дней, что заставляет деревья решений менять логику прогноза в зависимости от глобальной фазы рынка. При этом историческим наблюдениям присваивается вес, 
                затухающий по экспоненте с фактором 0.995, фокусируя модель на паттернах ближайшего прошлого (что важно для фокуса на текущих трендах, ведь изначально она получает на вход 500 торговых дней).
                """)
                st.write("""
                * **Гауссовская нормализация и волатильное таргетирование:** Для перевода агрегированного ранга предсказаний (ранжированных от 0 до 1) в вектор ожидаемых доходностей применяется обратное нормальное распределение (Гауссовский квантильный сдвиг). 
                Итоговый таргет масштабируется на краткосрочную рыночную волатильность, рассчитанную через 5-дневное EMA-окно, что позволяет автоматически занижать аппетит к риску в периоды шторма на рынке:
                """)
                st.latex(r"{r}_{i}=0.035\cdot \sigma _{5}\cdot \Phi ^{-1}(R_{i})")
                st.markdown("""
                * **L1-штраф за оборот:** Работает также как и в прочих институциональных стратегиях (<a href="#target_l1" target="_self">см. описание выше</a>).
                """, unsafe_allow_html=True)

            with st.expander("**Авторский оптимизатор (0.1 prototype)**  "
                "\n  Непараметрическая оптимизационная модель на базе сглаживания исторических сессий и адаптивного контура с динамическим переключением целевой функции.",
                expanded=st.session_state.expand_all_state):
                st.write("""
                * **Стратегическое квантование весов:** Полностью детерминированная математическая модель, отказывающаяся от точечного прогнозирования цен. Концепция базируется на многокритериальной выпуклой оптимизации ретроспективного 4-летнего пространства сессий, где каждому историческому дню присваивается индивидуальный вес ценности.
                * **Параболическое ядерное сглаживание:** На основе синтезированных весов дней и волатильности Паркинсона через параболическую функцию ядра конструируется вектор устойчивого математического ожидания роста (growth_incentive), выступающий главным драйвером максимизации доходности портфеля:
                """)
                st.write("""
                Интеллектуальная собственность (контур весов дней): Конкретный математический алгоритм нелинейного распределения весов дней представляет собой закрытую авторскую разработку. Данный контур сейчас активно исследуется и тестируется для дальнейшего улучшения прогностической стабильности результатов.
                """)
                st.markdown("""
                * **L1-штраф за оборот:** Работает также как и в прочих институциональных стратегиях (<a href="#target_l1" target="_self">см. описание выше</a>).
                """, unsafe_allow_html=True)

        if st.button("Раскрыть/скрыть весь список", key="toggle_expand_btn", use_container_width=True):
            st.session_state.expand_all_state = not st.session_state.expand_all_state
            st.rerun()

    render_slide_1()

# ----------------- СЛАЙД 2 (Индекс 1) -----------------
elif idx_active == 1:
    st.title("Историческое моделирование траекторий капитала на горизонте 2015–2026 гг.")


    @st.fragment
    def render_slide_2():
        if "scen_mode" not in st.session_state:
            st.session_state.scen_mode = "Lumpsum (1 млн руб)"

        scen_options = ["Lumpsum (1 млн руб)", "DCA (Регулярные взносы)", "FIRE (Жизнь на капитал)",
                        "Lumpsum (100 млн руб)"]
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
        fond_key = "100" in selected_scen
        is_lump_mode = "Lumpsum" in selected_scen
        initial_cap = 100_000_000.0 if fond_key else 1_000_000.0

        if "100 млн" in selected_scen:
            file_path = os.path.join(DATA_DIR, "fond.parquet")
            suffix = "Lump"
        else:
            file_path = os.path.join(DATA_DIR, "human.parquet")
            suffix = "Lump" if "Lumpsum" in selected_scen else ("DCA" if "DCA" in selected_scen else "FIRE")

        df_equity, precomputed_labels, df_pct_all = load_and_calculate_equity(
            file_path, suffix, initial_cap, is_lump_mode
        )

        if is_lump_mode:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.7, 0.3])
        else:
            fig = go.Figure()

        for col in df_equity.columns:
            if col.endswith(f"_{suffix}") or col.endswith(f"_{suffix}_tf") or col.startswith(f"Inflation_Benchmark_{suffix}"):
                if col.startswith("Inflation_Benchmark"): base_name = "Inflation_Benchmark"
                else: base_name = col.split(f"_{suffix}")[0]

                color = STRATEGY_COLORS.get(base_name, '#000000')
                type_line = 'dash' if 'Inflation' in col else 'solid'
                leg_label = precomputed_labels.get(col, col)

                if not col.endswith("_tf"):
                    trace_line = go.Scattergl(
                        x=df_equity.index,
                        y=df_equity[col],
                        mode='lines',
                        name=leg_label,
                        legendgroup=col,
                        line=dict(color=color, width=2, dash=type_line),
                        hovertemplate='%{x|%Y-%m-%d}<br>Капитал: %{y:,.0f} руб<extra></extra>',
                    )

                    if is_lump_mode:
                        daily_pct = df_pct_all[col].dropna()
                        pct_vals = daily_pct.to_numpy()
                        abs_vals = np.abs(pct_vals)

                        y_transformed = np.where(
                            abs_vals > 0.01,
                            np.sign(pct_vals) * 0.01 * (np.log10(abs_vals / 0.01) + 1),
                            pct_vals
                        )

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
                    else:
                        fig.add_trace(trace_line)

        if mobile:
            fig.update_layout(
                height=420 if is_lump_mode else 380,
                margin=dict(l=10, r=10, t=25, b=10),
                dragmode=False,
                hovermode="closest",
                
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    xanchor="center",
                    y=-0.05 if is_lump_mode else -0.15,
                    x=0.5,
                    font=dict(size=10),
                    bgcolor='rgba(255, 255, 255, 0.6)'
                )
            )
            fig.update_xaxes(fixedrange=True)
            fig.update_yaxes(fixedrange=True)
        else:
            fig.update_layout(
                height=780 if is_lump_mode else 620,
                hovermode="x unified",
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
                hovermode="x unified")

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'showTips': False})

        if is_lump_mode:
            st.markdown(f"""* **{selected_scen}:** Начальный капитал {"100 000 000" if fond_key else "1 000 000"} руб. Инвестирован единовременно на начальной точке трека без последующих пополнений или изъятий 
            для оценки эффективности стратегий на уровне {"крупного фонда" if fond_key else "частного инвестора"}.""")
            st.markdown("""
            * **Описание графика:** 
            <span style="text-indent: 20px; display: inline-block;">  1. Верхняя часть отображает динамику стоимости чистых активов портфеля с учетом обязательного автоматического реинвестирования.
            <span style="text-indent: 20px; display: inline-block;">  2. Нижняя часть показывает доходности каждой торговой сессии на логарифмической шкале без учета удержания ежегодного налога на прирост капитала.</span>
            """, unsafe_allow_html=True)
        else:
            if "DCA" in selected_scen: st.markdown(f"* **{selected_scen}:** Начальный капитал 50 000 руб. Ежемесячное пополнение на базовую сумму 50 000 руб., которая динамически индексируется на величину накопленной инфляции.")
            elif "FIRE" in selected_scen: st.markdown(f"* **{selected_scen}:** Начальный капитал 6 000 000 руб. Ежемесячное изъятие средств для жизни на капитал составляет 60 000 руб., которая динамически индексируется на величину накопленной инфляции.")
            st.markdown("* **Описание графика:** Отображает динамику стоимости чистых активов портфеля с учетом обязательного автоматического реинвестирования.")


    render_slide_2()
# ----------------- СЛАЙД 3 (Индекс 2) -----------------
elif idx_active == 2:
    st.title("Статистический анализ избыточной доходности и инвестиционной эффективности")


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

        zoom_izraelson = False
        if metric == 'sortino':
            zoom_izraelson = st.checkbox(
                "🔬 Оптимизировать масштаб оси под модификацию Израильсона (приблизить отрицательные хвосты)",
                value=False
            )

        fig_box = px.box(
            df_runs,
            x='Strategy',
            y=metric,
            color='Strategy',
            color_discrete_map=STRATEGY_COLORS,
            title=f"Распределение метрики: {metrics_map[metric]}"
        )

        if mobile:
                fig_box.update_layout(
                    height=340,                          # Делаем чуть ниже, так как текста нет
                    margin=dict(l=10, r=10, t=15, b=40), # Максимум места под сам график
                    dragmode=False,
                    hovermode="closest",
                    
                    # Названия осей на телефонах просто СТИРАЕМ, чтобы освободить место
                    xaxis_title="", 
                    yaxis_title="",
                    
                    # Легенду оставляем включенной, но без лишних рамок
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.08,                         # Аккуратно прижимаем снизу к оси
                        xanchor="center",
                        x=0.5,
                        font=dict(size=10)
                    )
                )
                fig_box.update_xaxes(fixedrange=True)
                fig_box.update_yaxes(fixedrange=True)
        else:
            fig_box.update_layout(
                height=550,
                xaxis_title="Модель оптимизации",
                yaxis_title=metrics_map[metric] if 'metric' in locals() else "",
                showlegend=True,
                legend=dict(
                    x=1.02,                          # На ПК выносим легенду направо
                    y=1.0,
                    xanchor='left',
                    yanchor='top'
                )
            )

        if metric == 'sortino' and zoom_izraelson:
            fig_box.update_layout(yaxis=dict(range=[df_runs['sortino'].min() - 0.01, 0.01]))
            st.caption(f"ℹ️ *Используется верхняя граница в {0.01} для корректного отображения хвоста распределения (искаженным из-за коррективы Израильсона).*")

        st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': False, 'showTips': False})
        if metric == 'cagr':
            st.markdown("""
            * **Экономический смысл:** Среднегодовой темп роста чистой стоимости активов с учетом сложного процента.
            * **Математический аппарат:** 
            """)
            st.latex(r"\text{CAGR}=\left(\frac{S_{T}}{S_{0}}\right)^{\frac{252}{T}}-1")
            st.info("💡На графике величина приведена к процентному масштабу.")
        elif metric == 'sortino':
            st.markdown("""
            * **Экономический смысл:** Коэффициент Сортино, оценивающий величину избыточной доходности стратегии над безрисковой ставкой (LQDT в данном исследовании) в расчете на единицу волатильности исключительно отрицательных хвостовых отклонений относительно 0.
            * **Математический аппарат:** Расчет опирается на вектор избыточных сессионных доходностей безналогового трека:
            """)
            st.latex(r"\sigma _{\text{downside}}=\sqrt{\frac{252}{T}\sum _{t=1}^{T}\text{min}^{2} (0,r_t - r_{\text{LQDT}, t})}")
            st.latex(r"\text{Sortino}=\begin{cases}\frac{\text{CAGR}_{\text{excess}}}{\sigma _{\text{downside}}},&\text{если CAGR}_{\text{excess}}>0\\ \text{CAGR}_{\text{excess}}\cdot \sigma _{\text{downside}},&\text{если CAGR}_{\text{excess}}\le 0\end{cases}, где \ \text{CAGR}_{\text{excess}} = \text{CAGR}_{\text{taxless}} - \text{CAGR}_{\text{LQDT}}")
            st.info("💡Модифицированное нелинейное произведение при отрицательной избыточной доходности (модификация Израильсона) используется в исследовании - тк корректно ранжирует, хотя и сжимает отрицательные «хвосты».")
        elif metric == 'jenser':
            st.markdown("""
            * **Экономический смысл:** Альфа Дженсена, измеряющая абсолютную избыточную доходность стратегии относительно рыночного бенчмарка (MCFTRR в данном исследовании), скорректированную на величину системного рыночного риска портфеля (β).
            * **Математический аппарат:** Коэффициент системного риска портфеля (β) рассчитывается через отношение ковариации избыточных доходностей стратегии и индекса к дисперсии индекса на безналоговом треке:
            """)
            st.latex(r"\beta =\frac{\mathrm{Cov}\>(r_{t}-r_{\text{LQDT},t},\,\,r_{\text{MCFTRR},t}-r_{\text{LQDT},t})}{\mathrm{Var}\>(r_{\text{MCFTRR},t}-r_{\text{LQDT},t})+10^{-12}}")
            st.markdown("""
            Итоговое значение Альфы формируется на базе аннуализированных геометрических доходностей:
            """)
            st.latex(r"\alpha =\text{CAGR}_{\text{taxless}}- [ \text{CAGR}_{\text{LQDT}}+\beta \cdot (\text{CAGR}_{\text{MCFTRR}}-\text{CAGR}_{\text{LQDT}}) ]")
            st.info("💡На графике величина приведена к процентному масштабу.")


    render_slide_3()

# ----------------- СЛАЙД 4 (Индекс 3) -----------------
elif idx_active == 3:
    st.title("Оценка плотности распределения хвостовых рисков портфеля")


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

        fig_violin.update_traces(
            marker=dict(
                size=2
            ),
            spanmode='hard'
        )

        st.plotly_chart(fig_violin, use_container_width=True, config={'displayModeBar': False, 'showTips': False})
        if risk_metric == 'max_drawdown':
            st.markdown("""
            * **Экономический смысл:** Максимальная глубина падения стоимости портфеля от локального исторического пика до последующего минимума. Показывает предельный бумажный убыток, который пришлось пересидеть инвестору на всем временном интервале.
            * **Математический аппарат:** Расчет текущей просадки портфеля и фиксация ее экстремума:
            """)
            st.latex(r"\text{MaxDD}=\min _{t}\left( \frac{S_{t}}{\max _{\tau \le t}S_{\tau }} - 1\right)")
            st.info("💡На графике величина приведена к процентному масштабу.")
        elif risk_metric == 'max_daily_drop':
            st.markdown("""
            * **Экономический смысл:** Предельный дневной убыток портфеля за одну отдельную торговую сессию. Измеряет устойчивость стратегии к мгновенным ценовым гэпам («черным лебедям») внутри одного торгового дня.
            * **Математический аппарат:** Поиск абсолютного минимума в векторе ежедневных доходностей портфеля:
            """)
            st.latex(r"\text{MaxDailyDrop}=\min \left(\min _{t}r_{t}, 0\right)")
            st.info("💡На графике величина приведена к процентному масштабу.")
        elif risk_metric == 'ulcer':
            st.markdown("""
            * **Экономический смысл:** Индекс Ульцера оценивает совокупный психологический и временной износ капитала — то есть как глубоко и как долго портфель находился в состоянии просадки.
            * **Математический аппарат:** Рассчитывается как среднеквадратичное значение ежедневных просадок от локального пика портфеля:
            """)
            st.latex(r"\text{UI}=-\sqrt{\frac{1}{T}\sum_{t=1}^T\left(1 - \frac{S_{t}}{\max _{\tau \le t}S_{\tau }}\right)^2}")
            st.info("💡На графике величина приведена к процентному масштабу.")


    render_slide_4()

# ----------------- СЛАЙД 5 (Индекс 4) -----------------
elif idx_active == 4:
    st.title("Двумерный анализ робастности стратегий в координатах доходности и риска")


    @st.fragment
    def render_slide_5():
        if "active_contour_mode" not in st.session_state:
            st.session_state.active_contour_mode = "contour_a"

        contours_map = {
            'contour_a': "Контур А: Классический (CAGR vs Max Drawdown)",
            'contour_b': "Контур Б: Продвинутый (Jensen's Alpha vs Ulcer Index)"
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

        fig_scatter = go.Figure()

        for strat_name, color in STRATEGY_COLORS.items():
            strat_df = df_runs[df_runs['Strategy'] == strat_name]

            if contour_mode == 'contour_a':
                x_vals = np.abs(strat_df['max_drawdown'].to_numpy())
                y_vals = strat_df['cagr_taxless'].to_numpy() - strat_df['cagr_lqdt'].to_numpy()

                x_title, y_title = "Максимальная историческая просадка (MaxDD, %)", "Премия за риск (избыточный CAGR, % годовых)"
                title_text = "Анализ классической эффективности торговых систем"
            else:
                x_vals = np.abs(strat_df['ulcer'].to_numpy())
                y_vals = strat_df['jenser'].to_numpy()

                x_title, y_title = "Индекс Ульцера (Временной износ фонда, %)", "Чистая Альфа Дженсена (Jensen's Alpha, % годовых)"
                title_text = "Анализ регуляторного качества и устойчивости стратегий"

            fig_scatter.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                name=strat_name,
                legendgroup=strat_name,
                showlegend=False,
                marker=dict(color=color, size=3),
                hovertemplate=f"<b>{strat_name}</b><br>{x_title}: %{{x:.2f}}%<br>{y_title}: %{{y:.2f}}%<extra></extra>"
            ))

            median_x = np.median(x_vals)
            median_y = np.median(y_vals)

            fig_scatter.add_trace(go.Scattergl(
                x=[median_x], y=[median_y],
                mode='markers',
                name=f"{strat_name}",
                legendgroup=strat_name,
                showlegend=True,
                marker=dict(color=color, size=12, symbol='square', line=dict(color='black', width=1)),
                hovertemplate=f"<b>Центр масс: {strat_name}</b><br>Медиана X: {median_x:.2f}%<br>Медиана Y: {median_y:.2f}%<extra></extra>"
            ))

            vals, vecs = np.linalg.eigh(np.cov(x_vals, y_vals))
            t = np.linspace(0, 2 * np.pi, 100)
            pts = (vecs * (2.45 * np.sqrt(vals))) @ np.vstack([np.cos(t), np.sin(t)])
            ex, ey = pts[0] + x_vals.mean(), pts[1] + y_vals.mean()

            fig_scatter.add_trace(go.Scatter(
                x=ex,
                y=ey,
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

        st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False, 'showTips': False})

        if contour_mode == "contour_a":
            st.markdown("""
            **Координаты:** Избыточная доходность (разность CAGR относительно LQDT) к пиковому риску (максимальная просадка).
            """)
        elif contour_mode == "contour_b":
            st.markdown("""
            **Координаты:** Эффективность относительно рынка (альфа Дженсена) к временному износу капитала (индекс Ульцера).
            """)
        st.markdown(f"""
        **Визуальный базис:**  \n  
        * **Медианный центр масс (◼):** Квадратный маркер фиксирует геометрический центр распределения результатов по независимым медианам осей. Применение медианы изолирует центр распределения от смещений, вызванных единичными экстремальными рыночными сессиями.
        * **Доверительный эллипс распреления (Пунктир):** Задает 95%-ю границу двумерной локализации исходов во всех инвестиционных прогонах. Направление наклона, геометрия и площадь эллипса определяются собственными векторами и значениями выборочной ковариационной матрицы парных метрик, масштабированными через критический квантиль распределения Чи-квадрат.
        """)


    render_slide_5()

# ----------------- СЛАЙД 6 (Индекс 5) -----------------
elif idx_active == 5:
    st.title("Комплексная оценка инвестиционной привлекательности и зон доминирования моделей")


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

        st.markdown(f"""
        **Математический базис контура:** Для кросс-секционного сопоставления результатов на скользящих окнах избыточная доходность стратегии нормируется на принятый хвостовой риск, возведенный в степень жесткости (p). Формула K-оценки автоматически перестраивается в зависимости от выбранного аналитического контура:
        """)
        if contour == "A":
            st.latex(r"K_n=\frac{\text{CAGR}_{\text{taxless},n}-\text{CAGR}_{\text{lqdt},n}}{\text{MaxDD}_n^p}")
            st.markdown(f"""
            Оценивает премию за риск (разность геометрического темпа роста безналогового трека и безрискового кэш-бенчмарка LQDT) на единицу пикового убытка.
            """)
        elif contour == "B":
            st.latex(r"K_n=\frac{\text{Jensen}_n}{\text{Ulcer}_n^p}")
            st.markdown(f"""
            Оценивает аннуализированную Альфу Дженсена (избыточную доходность, очищенную от системного рыночного риска портфеля) на единицу износа капитала.
            """)

        model = st.radio("Модель", target_models, horizontal=True)

        has_shock = (df["start_date"] < "2022-03-24") & (df["end_date"] >= "2022-03-24")
        if regime == "С 24.03.2022":
            df = df[has_shock]
        elif regime == "Без 24.03.2022":
            df = df[~has_shock]

        pivot_df = df.pivot_table(
            index=key,
            columns="Strategy",
            values=["cagr_taxless", "jenser", "max_drawdown", "ulcer"],
            aggfunc="first"
        )

        cagr = pivot_df["cagr_taxless"]
        jen = pivot_df["jenser"]
        dd = pivot_df["max_drawdown"].abs()
        ui = pivot_df["ulcer"].abs()
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

        st.subheader("Частотный анализ робастности модели против эталона рыночного шума")
        fig = go.Figure()
        start = {n: -(r[0] + r[1] + r[2] / 2) for n, r in rows.items()}
        for i, (cat, color) in enumerate(
                zip(["Уверенное поражение", "Поражение", "Ничья", "Победа", "Уверенная победа"],
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
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'showTips': False})

        st.markdown(f"""
        * **Природа эталона шума:** Верхняя строка формируется как разность коэффициентов эффективности K между базовыми пассивными стратегиями Uniform 1/N и Random Monkey. 
        Использование данных бенчмарков обусловлено идентичностью их пассивной инвестиционной логики, что позволяет изолировать чистый фон стохастических колебаний Мосбиржи на скользящих интервалах.
        * **Итоговые оценки:** Порог «Победа/Поражение» фиксирует выход дельты стратегий за пределы {low}-го перцентиля распределения шума. Порог «Уверенная победа/Поражение» определяется 99.9-м перцентилем шума, 
        подтверждая статистически значимое превосходство алгоритма на фоне шума. Строка **1vsAll** представляет собой агрегированный стресс-тест, фиксирующий долю окон, на которых модель одновременно опередила абсолютно все доступные бенчмарки выборки.
        """)

        st.subheader("Границы стратегического доминирования для разных профилей инвесторов")
        st.markdown(f"""
        **Математический базис контура:** Анализ устойчивости инвестиционного превосходства в рамках максимизации функции полезности:
        """)
        st.latex(r"F = \Delta \text{Return} - \lambda \cdot \Delta \text{Risk}")

        grid = {
            "сверх-агрессивного": np.linspace(0, 0.5, 11),
            "агрессивного": np.linspace(0.5, 1.25, 11),
            "сбалансированного": np.linspace(1.25, 2.5, 11),
            "консервативного": np.linspace(2.5, 5, 11),
            "сверх-консервативного": np.linspace(5, 20, 11),
        }

        html_table = """
        <table style="width:100%; border-collapse: collapse; font-family: inherit; font-size: 14px; table-layout: fixed;">
            <thead>
                <tr style="background-color: rgba(49, 51, 63, 0.05); border-bottom: 2px solid rgba(49, 51, 63, 0.1);">
                    <th style="width: 20%; text-align: left; padding: 10px;">Стратегия</th>
                    <th style="width: 65%; text-align: left; padding: 10px;">Вердикт</th>
                    <th style="width: 15%; text-align: center; padding: 10px;">Гарантированная доля побед, %</th>
                </tr>
            </thead>
            <tbody>
        """

        for b in bench:
            wins_all = 1.0
            verdict_pos = []
            verdict_med = []
            verdict_neg = []

            for inv, num in grid.items():
                dc = (ret[model] - ret[b]).reindex(K.index).to_numpy()
                dr = (risk[model] - risk[b]).reindex(K.index).to_numpy()
                wins = ((dc[None, :] - num[:, None] * dr[None, :]) > 0).mean(axis=1)
                above = wins > 0.5
                if above.all(): verdict_pos.append(inv)
                elif not above.any(): verdict_neg.append(inv)
                else: verdict_med.append((inv, num[np.argmax(above[1:] != above[:-1]) + 1]))
                wins_all = min(wins_all, wins.min())

            verdicts = []

            if len(verdict_pos) == 5: verdicts.append("Лучше для любого типа инвестора;")
            elif len(verdict_neg) == 5: verdicts.append("Хуже для любого типа инвестора;")
            else:
                if verdict_pos:
                    verdicts.append(f"Лучше для {', '.join(verdict_pos)};")
                if verdict_med:
                    for tp, val in verdict_med:
                        verdicts.append(f"Сопоставим для {tp} при λ* ≈ {val:.2f};")
                if verdict_neg:
                    verdicts.append(f"Хуже для {', '.join(verdict_neg)};")

            row_count = len(verdicts)
            share_val = round(100 * wins_all, 2)

            for i, text in enumerate(verdicts):
                html_table += ("<tr style='border-bottom: 1px solid rgba(49, 51, 63, 0.08);'>")

                if i == 0: html_table += f'<td rowspan="{row_count}" style="padding: 12px 10px; font-weight: bold; vertical-align: middle; background: white; border-right: 1px solid rgba(49, 51, 63, 0.05);">{b}</td>'

                html_table += f'<td style="padding: 10px; white-space: normal; word-wrap: break-word; vertical-align: middle;">{text}</td>'

                if i == 0: html_table += f'<td rowspan="{row_count}" style="padding: 12px 10px; text-align: center; vertical-align: middle; background: white;">{share_val:.2f}</td>'

                html_table += "</tr>"

        html_table += "</tbody></table>"
        st.markdown(html_table, unsafe_allow_html=True)
        st.markdown("""
        * **Экономическая классификация инвестиционных профилей:**  \n  
        <span style="text-indent: 32px; display: inline-block;">   **Сверх-агрессивный (λ ≤ 0.5):** Фокус на чистой Альфе и CAGR без штрафа за просадки.  \n  
        <span style="text-indent: 32px; display: inline-block;">   **Агрессивный (0.05 < λ ≤ 1.25):** Высокий приоритет доходности при умеренном учете рисков.  \n  
        <span style="text-indent: 32px; display: inline-block;">   **Сбалансированный (1.25 < λ ≤ 2.5):** Равновесная оценка темпа роста и хвостовых шоков портфеля.  \n  
        <span style="text-indent: 32px; display: inline-block;">   **Консервативный (2.5 < λ ≤ 5.0):** Повышенный штраф за пиковые убытки капитала.  \n  
        <span style="text-indent: 32px; display: inline-block;">   **Сверх-консервативный (λ > 5.0):** Максимальное неприятие риска и времени нахождения под стрессом.  \n  
        """, unsafe_allow_html=True)
        st.markdown("""
        * **Статистический смысл таблицы:**  \n  
        <span style="text-indent: 32px; display: inline-block;">  **Колонка «Вердикт»:** Консолидирует дискретные решения по каждому профилю риска. Выводит перечень профилей, 
        где модель безусловно выигрывает бенчмарк (доля побед по окнам >= 50%), проигрывает ему или оказывается сопоставимой с фиксацией точной критической точки перелома знака полезности (λ*), 
        при которой доминирование меняется на противоположное.  \n  
        <span style="text-indent: 32px; display: inline-block;">  **Колонка «Гарантированная доля побед»:** Отображает абсолютный минимум устойчивости торговой системы. 
        Фиксирует наименьшую долю успешных исходов модели среди всех доступных окон текущей выборки при самом неблагоприятном (максимально штрафующем модель) значении параметра из всех пяти оцениваемых профилей риска.
        """, unsafe_allow_html=True)


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
        } else if (e.code === "Space") {
            e.preventDefault();
            const buttons = doc.querySelectorAll("button");
            for (let btn of buttons) {
                if (btn.innerText.includes("скрыть")) {
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
    st.query_params["slide"] = st.session_state.current_slide_index + 1
    st.rerun()

if next_clicked:
    st.session_state.current_slide_index += 1
    st.query_params["slide"] = st.session_state.current_slide_index + 1
    st.rerun()
