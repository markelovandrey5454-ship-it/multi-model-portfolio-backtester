import pandas as pd
from data_repository import LocalCSVStorage
import numpy as np

class DataCleaner:
    def __init__(self, storage: LocalCSVStorage):
        self.storage = storage

    def build_cleaned_market_data(self, assets_config: dict, target_start: str, delist_history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        close_series_dict = {}
        high_series_dict = {}
        low_series_dict = {}

        for asset_name, stages in assets_config.items():
            df = self.storage.get_market_data(asset_name=asset_name, stages=stages, target_start=target_start)
            if not df.empty:
                close_series_dict[asset_name] = df[f"{asset_name}_close"]
                high_series_dict[asset_name] = df[f"{asset_name}_high"]
                low_series_dict[asset_name] = df[f"{asset_name}_low"]

        price_matrix = pd.concat(close_series_dict.values(), axis=1,keys=close_series_dict.keys()).sort_index().replace(0.0, np.nan)
        high_matrix = pd.concat(high_series_dict.values(), axis=1, keys=high_series_dict.keys()).sort_index().replace(0.0, np.nan)
        low_matrix = pd.concat(low_series_dict.values(), axis=1, keys=low_series_dict.keys()).sort_index().replace(0.0,np.nan)

        if 'Денежный рынок(REPO)' in price_matrix.columns and 'Денежный рынок(LQDT)' in price_matrix.columns:
            price_matrix['Денежный рынок(REPO)'] = price_matrix['Денежный рынок(REPO)'].ffill()
            first_lqdt_date = price_matrix['Денежный рынок(LQDT)'].first_valid_index()
            t_ipo = price_matrix.index.get_loc(first_lqdt_date)
            base_lqdt_price = price_matrix.loc[first_lqdt_date, 'Денежный рынок(LQDT)']

            repo_rates = price_matrix['Денежный рынок(REPO)'].to_numpy()
            synthetic_prices = np.zeros(len(price_matrix))
            synthetic_prices[t_ipo] = base_lqdt_price

            for t in range(t_ipo - 1, -1, -1):
                daily_repo_rate = (1.0 + (repo_rates[t] / 100.0)) ** (1.0 / 252.0)
                synthetic_prices[t] = synthetic_prices[t + 1] / daily_repo_rate

            df_synthetic = pd.Series(synthetic_prices[:t_ipo], index=price_matrix.index[:t_ipo])
            price_matrix['Денежный рынок(LQDT)'] = price_matrix['Денежный рынок(LQDT)'].combine_first(df_synthetic)
            high_matrix['Денежный рынок(LQDT)'] = high_matrix['Денежный рынок(LQDT)'].combine_first(df_synthetic)
            low_matrix['Денежный рынок(LQDT)'] = low_matrix['Денежный рынок(LQDT)'].combine_first(df_synthetic)

        for col in price_matrix.columns:
            f_idx = price_matrix[col].first_valid_index()
            if f_idx is not None:
                price_matrix.loc[f_idx:, col] = price_matrix.loc[f_idx:, col].ffill()
                high_matrix.loc[f_idx:, col] = high_matrix.loc[f_idx:, col].ffill()
                low_matrix.loc[f_idx:, col] = low_matrix.loc[f_idx:, col].ffill()

        for ticker in price_matrix.columns:
            if ticker in ['Денежный рынок(REPO)', 'Денежный рынок(LQDT)']: continue
            p = price_matrix[ticker].to_numpy()
            for t in range(1, len(p)):
                if pd.isna(p[t]) or pd.isna(p[t - 1]): continue
                if p[t] / p[t - 1] <= 0.4 or p[t] / p[t - 1] >= 2.5:
                    for ratio in [1000.0, 100.0, 20.0, 10.0, 8.0, 3.0, 0.1, 0.01, 0.001, 0.0002]:
                        if p[t - 1] * 0.8 <= p[t] * ratio <= p[t - 1] * 1.2: break
                    else: ratio = p[t - 1] / p[t]

                    price_matrix.iloc[:t, price_matrix.columns.get_loc(ticker)] /= ratio
                    high_matrix.iloc[:t, high_matrix.columns.get_loc(ticker)] /= ratio
                    low_matrix.iloc[:t, low_matrix.columns.get_loc(ticker)] /= ratio

        returns_matrix = (price_matrix - price_matrix.shift(1)) / price_matrix.shift(1)

        div_matrix = pd.DataFrame(0.0, index=price_matrix.index, columns=price_matrix.columns)

        translate_cur = {'USD':'Доллар', 'EUR':'Евро'}

        for asset_name, stages in assets_config.items():
            if stages[0]['market'] != 'shares' or stages[0]['secid'] == 'LQDT': continue
            tickers = {st['secid'] for st in stages}
            df_div = self.storage.get_dividends_data(asset_name=asset_name, tickers=list(tickers))
            if df_div.empty: continue

            for idx, row in df_div.iterrows():
                if idx < price_matrix.index.min() or idx > price_matrix.index.max(): continue
                payout_date = div_matrix.index[div_matrix.index >= idx][0] if idx not in div_matrix.index else idx

                div_value, currency = float(row['value']), str(row['currency']).upper().strip()
                if currency in ['USD', 'EUR']:
                    fx_date = price_matrix.index[price_matrix.index <= idx][-1] if idx not in price_matrix.index else idx
                    rub_value = div_value * price_matrix.loc[fx_date, translate_cur[currency]]
                else:
                    rub_value = div_value

                price_date = price_matrix.index[price_matrix.index < idx][-1]
                rub_value *= 0.87
                high_matrix.loc[payout_date, asset_name] += rub_value
                low_matrix.loc[payout_date, asset_name] += rub_value
                div_matrix.loc[payout_date, asset_name] += rub_value / price_matrix.loc[price_date, asset_name]

        prev_close_matrix = price_matrix.shift(1)


        robust_high = np.maximum(high_matrix.to_numpy(), prev_close_matrix.to_numpy())
        robust_low = np.minimum(low_matrix.to_numpy(), prev_close_matrix.to_numpy())
        robust_low = np.where(robust_low == 0, 1e-8, robust_low)

        const_factor = 1.0 / (4.0 * np.log(2.0))
        vol_numpy = np.sqrt(const_factor * (np.log(robust_high / robust_low) ** 2))
        vol_matrix = pd.DataFrame(vol_numpy, index=price_matrix.index, columns=price_matrix.columns)

        for col in price_matrix.columns:
            first_valid_idx = price_matrix[col].first_valid_index()
            if first_valid_idx is not None:
                vol_matrix.loc[:first_valid_idx, col] = np.nan
                post_ipo_slice = vol_matrix.loc[first_valid_idx:, col].to_numpy()

                if col == 'Денежный рынок(LQDT)':
                    post_ipo_slice = np.where(np.isnan(post_ipo_slice) | (post_ipo_slice < 0.0001), 0.0001,
                                              post_ipo_slice)
                else:
                    post_ipo_slice = np.where(np.isnan(post_ipo_slice) | (post_ipo_slice == 0.0), 0.0005,
                                              post_ipo_slice)

                vol_matrix.loc[first_valid_idx:, col] = post_ipo_slice

        for m in [returns_matrix, div_matrix, high_matrix, low_matrix, vol_matrix]:
            m.drop(columns=['Денежный рынок(REPO)'], inplace=True)

        for delist_company in delist_history.index:
            delist_date = delist_history.loc[delist_company, "Date"]
            if delist_company in returns_matrix.columns:
                returns_matrix.loc[delist_date:, delist_company] = np.nan
                returns_matrix.loc[delist_date, delist_company] = -1.0
                vol_matrix.loc[delist_date:, delist_company] = np.nan
                if delist_history.loc[delist_company, "Currency"] in ['USD', 'EUR']:
                    fx_date = price_matrix.index[price_matrix.index <= delist_date][-1]
                    delist_history.loc[delist_company, "Amount"] *= price_matrix.loc[fx_date, translate_cur[delist_history.loc[delist_company, "Currency"]]]
                price_date = price_matrix.index[price_matrix.index < delist_date][-1]
                delist_history.loc[delist_company, "Amount"] /= price_matrix.loc[price_date, delist_company]

        delist_history.drop(columns=['Currency'], inplace=True)
        delist_history.to_csv('data/matrix/global_delist_panel.csv')

        return returns_matrix, div_matrix, vol_matrix