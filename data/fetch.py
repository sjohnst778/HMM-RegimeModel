# data/fetch.py
# Data retrieval layer — abstracted so the source (yfinance, Eikon, etc.)
# can be swapped without touching model or pipeline code.
#
# To add a new data source:
#   1. Implement a class inheriting from BaseDataProvider
#   2. Implement fetch_prices(), fetch_macro(), fetch_rates()
#   3. Set DATA_PROVIDER = "eikon" (or your provider name) in config.py
#
# Current providers:
#   "yfinance" — default, free, suitable for development
#   "eikon"    — Thomson Reuters Eikon (requires eikon package + API key)

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from config import (
    EQUITY_TICKERS, MACRO_TICKERS, DATA_START,
    DATA_PROVIDER,
)


# ── Base class ────────────────────────────────────────────────────────────────

class BaseDataProvider(ABC):
    """
    Abstract base class for data providers.
    All providers must implement these three methods.
    """

    @abstractmethod
    def fetch_prices(
        self,
        tickers: list[str],
        start: str,
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch adjusted closing prices for a list of tickers.

        Returns
        -------
        pd.DataFrame
            DatetimeIndex, columns = tickers, values = adjusted close.
            Forward-filled and back-filled.
        """
        ...

    @abstractmethod
    def fetch_macro(
        self,
        tickers: dict[str, str],
        start: str,
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch macro/credit proxy series.

        Parameters
        ----------
        tickers : dict mapping descriptive name -> provider ticker
                  e.g. {"hy_spread": "HYG", "rates": "^TNX"}

        Returns
        -------
        pd.DataFrame
            DatetimeIndex, columns = descriptive names.
        """
        ...

    @abstractmethod
    def fetch_rates(
        self,
        start: str,
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch interest rate series (10y yield, yield curve etc.)

        Returns
        -------
        pd.DataFrame with at least column 'rates_10y'.
        """
        ...


# ── yfinance provider ─────────────────────────────────────────────────────────

class YFinanceProvider(BaseDataProvider):
    """
    Free data provider using yfinance.
    Suitable for development and backtesting.
    Limitations: no intraday, limited history on some series, no credit spreads.
    """

    def fetch_prices(self, tickers, start, end=None):
        import yfinance as yf
        raw = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]]
            prices.columns = tickers
        return prices.ffill().bfill()

    def fetch_macro(self, tickers, start, end=None):
        import yfinance as yf
        ticker_list = list(tickers.values())
        raw = yf.download(ticker_list, start=start, end=end,
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]]
            prices.columns = ticker_list
        prices = prices.ffill().bfill()
        # Rename to descriptive names
        reverse_map = {v: k for k, v in tickers.items()}
        return prices.rename(columns=reverse_map)

    def fetch_rates(self, start, end=None):
        import yfinance as yf
        raw = yf.download("^TNX", start=start, end=end,
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            rates = raw["Close"].rename(columns={"^TNX": "rates_10y"})
        else:
            rates = raw[["Close"]]
            rates.columns = ["rates_10y"]
        return rates.ffill().bfill()


# ── Eikon provider ────────────────────────────────────────────────────────────

class EikonProvider(BaseDataProvider):
    """
    Thomson Reuters Eikon data provider.

    Requirements:
        pip install eikon
        Set EIKON_API_KEY in environment or config.py

    Advantages over yfinance:
        - Credit spreads (CDX, iTraxx) — key BOCPD signals
        - VIX term structure
        - Earnings revision data
        - Longer and cleaner history on European indices
        - Intraday data if needed

    Ticker format: Eikon RICs (e.g. ".SPX" not "^GSPC", "USBMK=RR" for 10y)

    To use: set DATA_PROVIDER = "eikon" in config.py and provide
    EIKON_EQUITY_RICS, EIKON_MACRO_RICS in config.py.
    """

    # Default RIC mappings — override in config.py as needed
    EQUITY_RIC_MAP = {
        "^GSPC":     ".SPX",
        "^FTSE":     ".FTSE",
        "^STOXX50E": ".STOXX50E",
        "^N225":     ".N225",
        "^HSI":      ".HSI",
        "^KS11":     ".KS11",   # KOSPI -- optional
        "^AXJO":     ".AXJO",   # ASX 200 -- optional
    }

    # RIC map is now driven by MACRO_TICKERS_EIKON in config.py
    # This property just provides a fallback if config is not available
    @property
    def MACRO_RIC_MAP(self):
        try:
            from config import MACRO_TICKERS_EIKON
            return MACRO_TICKERS_EIKON
        except ImportError:
            return {
                "hy_spread": "CDXHY5Y=GFI",
                "ig_spread": "CDXIG5Y=GFI",
                "rates":     "USBMK=RR",
                "vix":       ".VIX",
            }

    def __init__(self):
        try:
            import eikon as ek
            import os
            api_key = os.environ.get("EIKON_API_KEY") or _get_eikon_api_key()
            ek.set_app_key(api_key)
            self._ek = ek
        except ImportError:
            raise ImportError(
                "Eikon package not installed. Run: pip install eikon"
            )
        except Exception as e:
            raise RuntimeError(f"Eikon initialisation failed: {e}")

    def fetch_prices(self, tickers, start, end=None):
        rics = [self.EQUITY_RIC_MAP.get(t, t) for t in tickers]
        df, _ = self._ek.get_data(
            rics,
            fields=["TR.CLOSEPRICE.Date", "TR.CLOSEPRICE"],
            parameters={"SDate": start, "EDate": end or "0D"},
        )
        # Pivot and rename back to standard ticker names
        df = df.pivot(index="Date", columns="Instrument", values="Close Price")
        df.index = pd.to_datetime(df.index)
        reverse_map = {v: k for k, v in self.EQUITY_RIC_MAP.items()}
        df = df.rename(columns=reverse_map)
        return df[tickers].ffill().bfill()

    def fetch_macro(self, tickers, start, end=None):
        rics = [self.MACRO_RIC_MAP.get(k, k) for k in tickers.keys()]
        df, _ = self._ek.get_data(
            rics,
            fields=["TR.CLOSEPRICE.Date", "TR.CLOSEPRICE"],
            parameters={"SDate": start, "EDate": end or "0D"},
        )
        df = df.pivot(index="Date", columns="Instrument", values="Close Price")
        df.index = pd.to_datetime(df.index)
        ric_to_name = {v: k for k, v in self.MACRO_RIC_MAP.items()}
        df = df.rename(columns=ric_to_name)
        return df.ffill().bfill()

    def fetch_rates(self, start, end=None):
        ric = "USBMK=RR"
        df, _ = self._ek.get_data(
            [ric],
            fields=["TR.CLOSEPRICE.Date", "TR.CLOSEPRICE"],
            parameters={"SDate": start, "EDate": end or "0D"},
        )
        df = df.set_index("Date")[["Close Price"]].rename(
            columns={"Close Price": "rates_10y"}
        )
        df.index = pd.to_datetime(df.index)
        return df.ffill().bfill()


def _get_eikon_api_key() -> str:
    """Try to load Eikon API key from config."""
    try:
        from config import EIKON_API_KEY
        return EIKON_API_KEY
    except ImportError:
        raise RuntimeError(
            "EIKON_API_KEY not found. Set it in config.py or as environment variable."
        )


# ── Provider factory ──────────────────────────────────────────────────────────

def get_provider() -> BaseDataProvider:
    """
    Return the configured data provider instance.
    Controlled by DATA_PROVIDER in config.py.
    """
    if DATA_PROVIDER == "yfinance":
        return YFinanceProvider()
    elif DATA_PROVIDER == "eikon":
        return EikonProvider()
    else:
        raise ValueError(
            f"Unknown DATA_PROVIDER: '{DATA_PROVIDER}'. "
            f"Options: 'yfinance', 'eikon'"
        )


# ── Public API ────────────────────────────────────────────────────────────────
# These are the functions called by the rest of the codebase.
# All fetches go through the cache layer:
#   - Cache CSV is read first
#   - Last 6 months are always refreshed from the provider
#   - Combined data is saved back to cache
#   - Caller sees a single clean DataFrame

def fetch_equity_prices(
    start: str = DATA_START,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Fetch equity index prices using the cache layer + configured provider.

    Cache strategy: read CSV, refresh last 6 months from provider, merge, save.
    On first run (no cache): fetches full history from DATA_START.
    """
    from data.cache import fetch_multi_with_cache
    provider = get_provider()
    df = fetch_multi_with_cache(EQUITY_TICKERS, provider)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    # Apply end filter if specified
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def fetch_macro_prices(
    start: str = DATA_START,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Fetch macro/credit proxy prices using the cache layer + configured provider.
    Uses descriptive names (hy_spread, ig_spread, etc.) as column names.
    """
    from data.cache import fetch_multi_with_cache, fetch_with_cache, PRICE_COL
    provider = get_provider()

    # Fetch each macro ticker individually (they may have different start dates)
    frames = {}
    for name, ticker in MACRO_TICKERS.items():
        try:
            df = fetch_with_cache(ticker, provider)
            frames[name] = df[PRICE_COL]
        except Exception as e:
            print(f"  WARNING: macro ticker {ticker} ({name}) failed — {e}")

    if not frames:
        raise ValueError("No macro data fetched")

    combined = pd.DataFrame(frames)
    combined.index = pd.to_datetime(combined.index)
    combined.index.name = "date"
    combined = combined.ffill().bfill()
    if end:
        combined = combined[combined.index <= pd.Timestamp(end)]
    return combined


def fetch_rates(
    start: str = DATA_START,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch interest rate series using the cache layer."""
    from data.cache import fetch_with_cache, PRICE_COL
    provider = get_provider()
    df = fetch_with_cache("^TNX", provider)
    result = df[[PRICE_COL]].rename(columns={PRICE_COL: "rates_10y"})
    result.index = pd.to_datetime(result.index)
    result.index.name = "date"
    if end:
        result = result[result.index <= pd.Timestamp(end)]
    return result
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


if __name__ == "__main__":
    print(f"Active provider: {DATA_PROVIDER}")
    prices = fetch_equity_prices()
    print(f"Equity prices: {prices.shape}  {prices.index[0].date()} -> {prices.index[-1].date()}")
    macro = fetch_macro_prices()
    print(f"Macro prices:  {macro.shape}  {macro.index[0].date()} -> {macro.index[-1].date()}")
