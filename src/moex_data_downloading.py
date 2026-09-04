import requests
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')


class MoexISSClient:
    def __init__(self):
        self.base_url = "https://iss.moex.com/iss"
        self.headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}

    def _build_url(self, ticker: str, engine: str = None, market: str = None, board: str = None, is_div: bool = False) -> str:
        if is_div:
            return f"{self.base_url}/securities/{ticker}/dividends.json"

        return f"{self.base_url}/history/engines/{engine}/markets/{market}/boards/{board}/securities/{ticker}.json"

    def fetch_raw_market(self, engine: str, market: str, board: str, ticker: str, start_date: str, end_date: str) -> list:
        url = self._build_url(engine=engine, market=market, board=board, ticker=ticker)
        all_rows = []
        start_cursor = 0

        logging.info(f"Запуск скачивания из MOEX ISS: {ticker} ({start_date} -> {end_date})")

        while True:
            params = {
                'from': start_date,
                'till': end_date,
                'start': start_cursor,
                'iss.meta': 'off',
                'history.columns': 'TRADEDATE,CLOSE,HIGH,LOW'
            }

            response_json = None
            for attempt in range(3):
                try:
                    response = requests.get(url, params=params, headers=self.headers, timeout=15)
                    response.raise_for_status()
                    response_json = response.json()
                    break
                except (requests.RequestException, ValueError) as e:
                    wait_time = 2**(attempt + 1)
                    logging.warning(
                        f"Сбой сети для {ticker} (Попытка {attempt + 1}/3). Ждем {wait_time}с... Ошибка: {e}")
                    time.sleep(wait_time)

            if response_json is None:
                logging.error(f"Не удалось скачать данные для {ticker} после 3 попыток.")
                break

            if 'history' not in response_json or not response_json['history']['data']:
                break

            page_data = response_json['history']['data']

            for row in page_data:
                if len(row) > 3 and row[0] is not None and row[1] is not None and row[2] is not None and row[
                    3] is not None:
                    if float(row[1]) > 0 and float(row[2]) > 0 and float(row[3]) > 0:
                        all_rows.append([row[0], float(row[1]), float(row[2]), float(row[3])])

            if len(page_data) < 100:
                break

            start_cursor += 100
            time.sleep(0.05)

        logging.info(f"Завершено. Скачано строк из API для {ticker}: {len(all_rows)}")
        return all_rows

    def fetch_raw_dividends(self, ticker: str) -> list:
        url = self._build_url(ticker=ticker, is_div=True)
        params = {'iss.meta': 'off'}

        logging.info(f"Запрос дивидендов MOEX ISS для {ticker}")
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=15)
            response.raise_for_status()
            res_json = response.json()

            if 'dividends' in res_json and res_json['dividends']['data']:
                return res_json['dividends']['data']
            return []
        except Exception as e:
            logging.error(f"Сбой сети при запросе дивидендов {ticker}: {e}")
            return []