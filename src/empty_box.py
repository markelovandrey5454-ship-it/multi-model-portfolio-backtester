from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BasePortfolioStrategy(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def optimize_weights(self, historical_returns: pd.DataFrame, prev_weights: np.ndarray, volatility_history: pd.DataFrame = None) -> np.ndarray:
        pass

def decode_portfolio_to_bits(encoded_str, total_assets_count):
    BASE32_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUV"
    bit_array = []
    for char in encoded_str:
        if char not in BASE32_CHARS:
            continue
        val = BASE32_CHARS.index(char)
        chunk = []
        for _ in range(5):
            chunk.append(val & 1)
            val >>= 1
        bit_array.extend(reversed(chunk))
    return bit_array[:total_assets_count]