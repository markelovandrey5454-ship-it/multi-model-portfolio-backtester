# Multi-Model Portfolio Backtester & Optimization Engine

An industrial-grade quantitative framework for portfolio construction, asset allocation, and historical simulation on the Russian equity market (MOEX). The project implements a scalable pipeline to stress-test mathematical optimization models against heuristic, momentum, and classical financial benchmarks over an 11-year horizon.

> **Research Status:** Данный репозиторий представляет собой открытый инфраструктурный срез моей дипломной научно-исследовательской работы. Исходный код математических ядер финальных оптимизаторов (`Native model`, `Controlled-risk prototype`, `High-risk prototype`) инкапсулирован для защиты возможной интеллектуальной собственности.

---

---

## 📊 Implemented Strategies & Benchmarks

The framework evaluates the allocation engine across 5 open-source baselines and 4 proprietary mathematical models (accessible via unified structural interfaces):
1. **Uniform 1/N** — Naive baseline distributing capital equally among active assets.
2. **Random Monkey** — Stochastic weight generation simulating behavioral noise and random allocations.
3. **Markowitz Mean-Variance** — Classical optimization utilizing Ledoit-Wolf shrinkage covariance, L1 turnover constraints, and dynamic return fallbacks.
4. **Base CVaR** — Canonical Rockafellar-Uryasev Expected Shortfall optimization ($\beta=0.95$) constrained by strict regulatory asset weight limits (40% LQDT / 10% Equities).
5. **Momentum Rotator** — Deterministic trend-following model selecting Top-10 assets weighted linearly by 3-month cumulative returns.
6. **Personal Profile** — Adaptive heuristic expert system integrating age-based bond weighting, Gold/Silver ratios, and dynamic macro-cash allocation.
7. **Robust CVaR Series (v_old, v_new, 0.1)** — *[Closed Source]* Two-scale robust estimators resilient to market regime shifts, data noises, and structural fat-tailed shocks.

---

## 🏗️ Pipeline Architecture (`src/`)

The codebase strictly follows low-coupling modular software engineering principles:

* `src/empty_box.py` — Core abstract module housing the `BasePortfolioStrategy` unified interface and a custom decoder function converting 32-bit encoded strings into binary active asset matrices.
* `src/benchmarks.py` — Production implementation of all open-source financial baselines along with secure interface wrappers for proprietary closed-source model configurations.
* `src/data_cleaner.py` — High-performance data synchronization engine. Manages dividend tax adjustments with multi-currency scaling, resolves splits/consolidations, computes Parkinson volatility matrices, extends short liquidity tracks (*LQDT*) using REPO macro-rates, and flattens delisted history tracks.
* `src/data_repository.py` — Data management hub regulating automated caching and update directionalities (left/right append verification) for asset prices and scraped dividend aggregates.
* `src/moex_data_downloading.py` — Robust client interacting with the MOEX ISS API. Implements chunked data downloads (100 rows per request), throttling protections, connection lag fallbacks, and local indexing matching the exchange structure.
* `src/global_download.py` — Orchestrator entry point connecting asynchronous fetch channels into consolidated historical returns and volatility matrix outputs.
* `src/backtest_engine.py` — Structural simulator executing discrete asset rebalancing cycles, capital drift adjustments, and execution transaction slippages.
* `src/polytest.py` — Dedicated Monte Carlo simulation loop running batch tracks. Captures random timelines, active asset lists (32-bit encoded), and saves explicit distribution attributes (CAGR, Sortino relative to LQDT, Ulcer Index, Max Drawdown, Max Daily Drop) across models.
* `src/main_test.py` — Analytics script parsing pre-compiled Monte Carlo CSV run matrices to isolate risk-reward performance anomalies.
* `src/main.py` — Master execution script initiating continuous historical portfolio pathways (Lump-Sum, DCA, FIRE).
* `src/visualizer.py` — Graphics processing unit plotting KDE density boundaries, historical asset curves, stability distributions, and upcoming Time-Series Total Return validation tracking.

---

## 📈 Empirical Results & Visualizations

### 11-Year Historical Tracks (Continuous Macro-Simulations)
The engine evaluates multi-asset tracks across core wealth management paradigms: **Lump-Sum** deployments, regular Dollar-Cost Averaging (**DCA**), and Financial Independence capital withdrawal loops (**FIRE**).

### Monte Carlo Resilience Metrics (250 Random Regimes)
Detailed output tables mapping track parameters, historical drawdowns, and relative Sortino outperformance distributions are securely cataloged inside the `data/results/` and `images/current_runs/` directories:
* `boxplots_cagr_comparison.png` — Distribution of annualized return structures.
* `risk_density_plots.png` — Tail-risk probabilities mapped via KDE curves (Max Drawdown, Session Drops, Ulcer index).
* `stability_zones_comparison.png` — Stacked distribution boundaries proving model stability against classical benchmarks.

---

## 🚀 Local Installation & Quick Start

1. Clone the repository on your local machine:
   ```bash
   git clone https://github.com
   cd multi-model-portfolio-backtester
   ```
2. Setup dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run historical benchmark tracks:
   ```bash
   python src/main_test.py
   ```

---

## 💼 Career & Research Status
* **Author:** 4th-year Mathematical Faculty Student (Business Informatics, Saratov State University).
* **Domain:** Quantitative Finance / Portfolio Risk Engineering / Robust Convex Programming.
* **Availability:** Open for **remote internships / part-time remote roles (20-30 hours/week)** immediately, or **Full-Time Algorithmic Desk positions in Moscow** with an expected start/relocation in **July 2026** (post-graduation).
* **Contact:** Telegram: `@m_rk_l`

<details>
<summary>🌐 Посмотреть описание архитектуры проекта на русском языке</summary>

## Мультимодельный симулятор бэктестинга и портфельной оптимизации

Промышленный инфраструктурный пайплайн для автоматизации портфельного инжиниринга, очистки финансовых данных и исторического моделирования на рынке акций Московской Биржи (MOEX). 

### Ключевые компоненты архитектуры (`src/`):
* `empty_box.py` — Базовый абстрактный интерфейс стратегий и декодер 32-битных масок активов.
* `benchmarks.py` — Реализация 5 открытых бенчмарков (Марковиц Ледойта-Вулфа, Моментум-ротатор, 1/N, классический CVaR и авторский профиль) + защищенные интерфейсы приватных моделей.
* `data_cleaner.py` — Математический конвейер: учет НДФЛ 13% и валютной конвертации дивидендов, очистка сплитов, расчет волатильности по Паркинсону и ретроспективный синтез LQDT через РЕПО с ЦК.
* `data_repository.py` & `moex_data_downloading.py` — Кэширующий хаб и отказоустойчивый клиент к API ISS MOEX.
* `backtest_engine.py` — Симулятор петель ребалансировки с учетом рыночного дрейфа весов, комиссионных издержек и пауз в начислениях дивов/выкупов.
* `polytest.py` & `main_test.py` — Скрипты генерации и агрессивного анализа 250 случайных стресс-тестов Монте-Карло.

</details>