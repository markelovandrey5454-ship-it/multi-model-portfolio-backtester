import os
import pandas as pd
import datetime
import logging
from moex_data_downloading import MoexISSClient

today_str = datetime.date.today().strftime("%Y-%m-%d")

class LocalCSVStorage:
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.client = MoexISSClient()

    def _get_file_path(self, ticker: str, is_div: bool = False) -> str:
        suffix = "_div" if is_div else ""
        return os.path.join(self.cache_dir, f"{ticker}{suffix}.csv")

    def get_market_data(self, asset_name: str, stages: list, target_start: str) -> pd.DataFrame:
        file_path = self._get_file_path(ticker=asset_name, is_div=False)

        oldest_stage = stages[0]
        current_stage = stages[-1]
        cache_cols = ['Date', f"{asset_name}_close", f"{asset_name}_high", f"{asset_name}_low"]

        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            logging.info(f"{asset_name}: кэш пуст. Скачиваем полную историю: {target_start} -> {today_str}")
            stage_dfs = []

            for stage in stages:
                raw_data = self.client.fetch_raw_market(
                    engine=stage['engine'], market=stage['market'], board=stage['board'],
                    ticker=stage['secid'], start_date=target_start, end_date=today_str
                )
                if raw_data:
                    df_stage = pd.DataFrame(raw_data, columns=cache_cols)
                    df_stage['Date'] = pd.to_datetime(df_stage['Date'])
                    stage_dfs.append(df_stage.set_index('Date'))

            df_total = stage_dfs[0]
            for df_next in stage_dfs[1:]:
                df_total = df_total.combine_first(df_next)

            df_total = df_total[~df_total.index.duplicated(keep='first')].sort_index()
            df_total.to_csv(file_path)
            return df_total

        df_local = pd.read_csv(file_path, parse_dates=['Date']).set_index('Date')
        local_min = df_local.index.min().strftime("%Y-%m-%d")
        local_max = df_local.index.max().strftime("%Y-%m-%d")

        if local_max < today_str:
            logging.info(
                f"{asset_name}: докачка будущего по тикеру {current_stage['secid']} с {local_max} по {today_str}")
            raw_future = self.client.fetch_raw_market(
                engine=current_stage['engine'], market=current_stage['market'], board=current_stage['board'],
                ticker=current_stage['secid'], start_date=local_max, end_date=today_str
            )
            if raw_future:
                df_future = pd.DataFrame(raw_future, columns=cache_cols).set_index('Date')
                df_future.index = pd.to_datetime(df_future.index)
                df_local = df_local.combine_first(df_future)

        if target_start < local_min:
            logging.info(
                f"{asset_name}: расширение горизонта влево по тикеру {oldest_stage['secid']} с {target_start} по {local_min}")
            raw_past = self.client.fetch_raw_market(
                engine=oldest_stage['engine'], market=oldest_stage['market'], board=oldest_stage['board'],
                ticker=oldest_stage['secid'], start_date=target_start, end_date=local_min
            )
            if raw_past:
                df_past = pd.DataFrame(raw_past, columns=cache_cols).set_index('Date')
                df_past.index = pd.to_datetime(df_past.index)
                df_local = df_local.combine_first(df_past)

        df_local = df_local[~df_local.index.duplicated(keep='first')].sort_index()
        df_local.to_csv(file_path)
        return df_local

    def get_dividends_data(self, asset_name: str, tickers: list) -> pd.DataFrame:
        div_path = self._get_file_path(ticker=asset_name, is_div=True)
        stage_dfs = []
        for ticker in tickers:
            raw_data = self.client.fetch_raw_dividends(ticker=ticker)
            if raw_data:
                df_stage = pd.DataFrame(raw_data, columns=['secid', 'isin', 'registryclosedate', 'value', 'currencyid'])
                df_stage = df_stage[['registryclosedate', 'value', 'currencyid']].rename(
                    columns={'registryclosedate': 'Date', 'currencyid': 'currency'})
                df_stage['Date'] = pd.to_datetime(df_stage['Date'])
                stage_dfs.append(df_stage.set_index('Date'))

        if stage_dfs:
            df_net = stage_dfs[0]
            for df_next in stage_dfs[1:]:
                df_net = df_net.combine_first(df_next)

            if os.path.exists(div_path) and os.path.getsize(div_path) > 0:
                df_disk = pd.read_csv(div_path, parse_dates=['Date']).set_index('Date')
                df_disk.index = pd.to_datetime(df_disk.index)

                df_total = df_disk.combine_first(df_net)
            else:
                df_total = df_net

            df_total = df_total[~df_total.index.duplicated(keep='first')].sort_index()
            df_total.to_csv(div_path)
            return df_total
        return pd.DataFrame()
