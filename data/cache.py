# data/cache.py
# Local CSV cache layer for market data.
# Run from project root: python data/cache.py --backfill

import sys
from pathlib import Path
# Ensure project root is on the path when running this file directly
sys.path.insert(0, str(Path(__file__).parent.parent))
#
# Design:
#   - One CSV file per ticker in data/cache/
#   - On every fetch: read CSV, refresh last 6 months from provider, merge, save
#   - Provider (yfinance/Eikon) only ever called for recent data after initial load
#   - Rest of codebase sees a single clean DataFrame — no awareness of the split
#
# Usage:
#   python data/cache.py --backfill          # initial load of all configured tickers
#   python data/cache.py --backfill --ticker ^GSPC  # single ticker
#   python data/cache.py --status            # show cache coverage per ticker

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

CACHE_DIR     = Path("data/cache")
REFRESH_MONTHS = 6      # always re-fetch this many months from provider
DATE_COL      = "date"
PRICE_COL     = "close"


# ── Core cache operations ──────────────────────────────────────────────────────

def cache_path(ticker: str) -> Path:
    """Return the cache file path for a ticker."""
    safe = ticker.replace("^", "_").replace("/", "_").replace(".", "_")
    return CACHE_DIR / f"{safe}.csv"


def read_cache(ticker: str) -> pd.DataFrame | None:
    """
    Read cached price data for a ticker.

    Returns
    -------
    pd.DataFrame with DatetimeIndex and single column 'close', or None if no cache.
    """
    path = cache_path(ticker)
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=DATE_COL, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df.index.name = DATE_COL
    return df[[PRICE_COL]]


def write_cache(ticker: str, df: pd.DataFrame) -> None:
    """Write price data to cache CSV."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(ticker)
    df = df.copy()
    df.index.name = DATE_COL
    if PRICE_COL not in df.columns:
        raise ValueError(f"DataFrame must have a '{PRICE_COL}' column")
    df[[PRICE_COL]].sort_index().to_csv(path)


def merge_cache_with_fresh(
    cached: pd.DataFrame | None,
    fresh: pd.DataFrame,
    overlap_days: int = 10,
) -> pd.DataFrame:
    """
    Merge cached historical data with freshly fetched recent data.

    The fresh data wins on any overlap (handles late price revisions,
    corporate actions, etc.). An overlap_days buffer ensures no gaps
    at the join point.

    Parameters
    ----------
    cached       : historical data from cache (may be None)
    fresh        : recently fetched data from provider
    overlap_days : how many days of overlap to allow fresh data to overwrite

    Returns
    -------
    Combined DataFrame, sorted, duplicates removed (fresh wins).
    """
    if cached is None or cached.empty:
        return fresh.copy()

    # Trim cached data: keep everything before the fresh data's start
    # minus the overlap buffer
    fresh_start   = fresh.index[0]
    cache_cutoff  = fresh_start - pd.Timedelta(days=overlap_days)
    cached_trimmed = cached[cached.index < cache_cutoff]

    combined = pd.concat([cached_trimmed, fresh])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    return combined


def get_refresh_start(cached: pd.DataFrame | None, refresh_months: int = REFRESH_MONTHS) -> str:
    """
    Determine the start date for the provider fetch.

    Always re-fetches the last `refresh_months` months regardless of cache,
    to catch any late revisions. If cache is empty, fetches from DATA_START.
    """
    from config import DATA_START
    if cached is None or cached.empty:
        return DATA_START
    cutoff = datetime.today() - relativedelta(months=refresh_months)
    # Don't go earlier than DATA_START
    data_start = pd.Timestamp(DATA_START)
    return max(cutoff, data_start.to_pydatetime()).strftime("%Y-%m-%d")


# ── Cached fetch functions ─────────────────────────────────────────────────────

def fetch_with_cache(
    ticker: str,
    provider,
    refresh_months: int = REFRESH_MONTHS,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch a single ticker using the cache layer.

    1. Read existing cache
    2. Determine refresh window (last N months)
    3. Fetch fresh data from provider
    4. Merge and save
    5. Return full combined series

    Parameters
    ----------
    ticker         : ticker symbol (e.g. "^GSPC", "HYG")
    provider       : data provider instance (YFinanceProvider or EikonProvider)
    refresh_months : months to always re-fetch from provider

    Returns
    -------
    pd.DataFrame with DatetimeIndex and 'close' column.
    """
    cached      = read_cache(ticker)
    start       = get_refresh_start(cached, refresh_months)
    cache_rows  = len(cached) if cached is not None else 0

    if verbose:
        if cached is not None:
            print(f"  {ticker:15s}: cache={cache_rows} rows, "
                  f"refreshing from {start}")
        else:
            print(f"  {ticker:15s}: no cache, full fetch from {start}")

    # Fetch from provider
    try:
        fresh_multi = provider.fetch_prices([ticker], start=start)
        if fresh_multi.empty:
            if cached is not None and not cached.empty:
                if verbose:
                    print(f"  {ticker:15s}: provider returned no data, using cache only")
                return cached
            raise ValueError(f"No data available for {ticker}")
        fresh = fresh_multi.iloc[:, 0].rename(PRICE_COL).to_frame()
        fresh.index = pd.to_datetime(fresh.index)
    except Exception as e:
        if cached is not None and not cached.empty:
            if verbose:
                print(f"  {ticker:15s}: fetch failed ({e}), using cache only")
            return cached
        raise

    # Merge and save
    combined = merge_cache_with_fresh(cached, fresh)
    write_cache(ticker, combined)

    if verbose:
        print(f"  {ticker:15s}: total={len(combined)} rows, "
              f"{combined.index[0].date()} -> {combined.index[-1].date()}")

    return combined


def fetch_multi_with_cache(
    tickers: list[str],
    provider,
    refresh_months: int = REFRESH_MONTHS,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch multiple tickers using the cache layer, returning a combined DataFrame.

    Parameters
    ----------
    tickers : list of ticker symbols
    provider: data provider instance

    Returns
    -------
    pd.DataFrame with DatetimeIndex, one column per ticker (adjusted close).
    Forward-filled and back-filled.
    """
    frames = {}
    for ticker in tickers:
        try:
            df = fetch_with_cache(ticker, provider, refresh_months, verbose)
            frames[ticker] = df[PRICE_COL]
        except Exception as e:
            print(f"  WARNING: {ticker} failed — {e}")
            continue

    if not frames:
        raise ValueError("No data fetched for any ticker")

    combined = pd.DataFrame(frames)
    combined.index = pd.to_datetime(combined.index)
    combined = combined.ffill().bfill()
    return combined


# ── Cache management utilities ─────────────────────────────────────────────────

def cache_status() -> pd.DataFrame:
    """
    Print a summary of the current cache state for all tickers.

    Returns
    -------
    pd.DataFrame with columns: ticker, rows, start, end, age_days, size_kb
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(CACHE_DIR.glob("*.csv")):
        try:
            df   = pd.read_csv(path, index_col=0, parse_dates=True)
            rows = len(df)
            start = df.index[0].date() if rows else None
            end   = df.index[-1].date() if rows else None
            age   = (datetime.today().date() - end).days if end else None
            size  = path.stat().st_size / 1024
            # Reverse the ticker name sanitisation
            ticker = path.stem.replace("_", "^", 1) if path.stem.startswith("_") \
                     else path.stem.replace("_", "^")
            records.append({
                "file":     path.name,
                "rows":     rows,
                "start":    start,
                "end":      end,
                "age_days": age,
                "size_kb":  round(size, 1),
            })
        except Exception as e:
            records.append({"file": path.name, "error": str(e)})

    df = pd.DataFrame(records)
    return df


def clear_cache(ticker: str | None = None) -> None:
    """
    Clear cache for a specific ticker, or all tickers if ticker=None.
    Use with caution — requires a full re-fetch from provider.
    """
    if ticker:
        path = cache_path(ticker)
        if path.exists():
            path.unlink()
            print(f"Cleared cache for {ticker}")
        else:
            print(f"No cache found for {ticker}")
    else:
        count = 0
        for path in CACHE_DIR.glob("*.csv"):
            path.unlink()
            count += 1
        print(f"Cleared {count} cached files")


# ── Backfill utility ───────────────────────────────────────────────────────────

def backfill(ticker: str | None = None, verbose: bool = True) -> None:
    """
    Initial cache population — fetches full history for all configured tickers.
    Safe to re-run: only fetches what's missing.

    Parameters
    ----------
    ticker : specific ticker to backfill, or None for all configured tickers
    """
    from config import EQUITY_TICKERS, MACRO_TICKERS, DATA_START
    from data.fetch import get_provider

    provider = get_provider()
    print(f"Provider: {type(provider).__name__}")
    print(f"Data start: {DATA_START}")
    print()

    all_tickers = list(EQUITY_TICKERS)
    all_tickers += list(MACRO_TICKERS.values())

    if ticker:
        all_tickers = [ticker]
        print(f"Backfilling single ticker: {ticker}")
    else:
        print(f"Backfilling {len(all_tickers)} tickers: {all_tickers}")

    print()
    for t in all_tickers:
        try:
            fetch_with_cache(t, provider, refresh_months=REFRESH_MONTHS, verbose=verbose)
        except Exception as e:
            print(f"  ERROR {t}: {e}")

    print()
    print("Cache status after backfill:")
    status = cache_status()
    print(status.to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cache management utility")
    parser.add_argument("--backfill", action="store_true",
                        help="Populate cache for all configured tickers")
    parser.add_argument("--ticker", default=None,
                        help="Specific ticker (use with --backfill)")
    parser.add_argument("--status", action="store_true",
                        help="Show cache coverage summary")
    parser.add_argument("--clear", action="store_true",
                        help="Clear all cached data (requires re-fetch)")
    args = parser.parse_args()

    if args.backfill:
        backfill(ticker=args.ticker)
    elif args.status:
        print(cache_status().to_string(index=False))
    elif args.clear:
        confirm = input("Clear all cache? This requires a full re-fetch. [y/N] ")
        if confirm.lower() == "y":
            clear_cache()
    else:
        parser.print_help()
