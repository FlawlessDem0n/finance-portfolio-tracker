import math
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys

# --------- CONFIG (edit this) ----------
# Positions in SHARES. Example: {"AAPL": 50, "MSFT": 20, "TSLA": 10}
PORTFOLIO: Dict[str, float] = {
    "AAPL": 50,
    "MSFT": 20,
    "TSLA": 10,
    "GOOGL": 8
}
BENCHMARK = "^GSPC"         # S&P 500
LOOKBACK_DAYS = 365         # try 90, 180, 365, 730, etc.
RISK_FREE_ANNUAL = 0.00     # set to your T-bill proxy if desired (e.g., 0.05 = 5%)
# --------------------------------------

@dataclass
class Metrics:
    total_value: float
    weights: pd.Series
    daily_ret_series: pd.Series
    ann_return: float
    ann_vol: float
    sharpe: float
    beta: float

def fetch_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetch adjusted close prices for one or multiple tickers using yfinance.
    - auto_adjust=True => 'Close' is already adjusted.
    - forward/backward fill to avoid NaN wipeouts on short windows/holidays.
    - adds simple validations and friendly errors.
    """
    if not isinstance(tickers, (list, tuple)):
        tickers = [tickers]

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,   # 'Close' is adjusted
        group_by="column"   # MultiIndex when multiple tickers
    )

    if raw is None or raw.empty:
        raise ValueError(f"No price data for {tickers} between {start} and {end}.")

    # Multi vs single ticker handling
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise KeyError("Downloaded data has no 'Close' field.")
        close = raw["Close"].copy()
        # Keep order and only requested tickers
        close = close.loc[:, [t for t in tickers if t in close.columns]]
    else:
        if "Close" not in raw.columns:
            raise KeyError("Downloaded data has no 'Close' field.")
        close = raw[["Close"]].copy()
        name = tickers[0] if len(tickers) >= 1 else "TICKER"
        close.columns = [name]

    # Use fills instead of dropping (prevents empty frames on short lookbacks)
    close = close.ffill().bfill()

    if close.isna().all().all():
        raise ValueError(f"Fetched only NaN values for {tickers}. Try a different lookback/tickers.")

    # Drop any columns that remained fully NaN after fill (just in case)
    close = close.loc[:, ~close.isna().all(axis=0)]

    if close.empty:
        raise ValueError(f"All fetched prices are NaN for {tickers}. Try longer lookback or different tickers.")

    return close

def compute_metrics(prices: pd.DataFrame, shares: pd.Series, bench_prices: pd.Series) -> Metrics:
    prices = prices.ffill().bfill()
    bench_prices = bench_prices.ffill().bfill()

    latest_prices = prices.iloc[-1]
    position_values = latest_prices * shares
    total_value = float(position_values.sum())
    if total_value <= 0:
        raise ValueError("Total portfolio value is zero or negative. Check shares and tickers.")

    weights = (position_values / total_value).fillna(0.0)

    # Daily returns
    rets = prices.pct_change().dropna(how="all")
    rets = rets.loc[:, rets.notna().any(axis=0)]
    weights = weights.reindex(rets.columns).fillna(0.0)
    if weights.sum() > 0:
        weights = weights / weights.sum()

    port_daily = (rets * weights).sum(axis=1)

    bench_daily = bench_prices.pct_change().dropna()
    aligned = pd.concat([port_daily, bench_daily], axis=1, join="inner")
    aligned.columns = ["port", "bench"]

    trading_days = 252
    ann_vol = float(port_daily.std() * math.sqrt(trading_days)) if len(port_daily) > 1 else float("nan")

    if len(port_daily) > 0:
        cum_return = (1 + port_daily).prod() - 1
        years = len(port_daily) / trading_days
        ann_return = float((1 + cum_return) ** (1 / years) - 1) if years > 0 else float("nan")
    else:
        ann_return = float("nan")

    rf_daily = (1 + RISK_FREE_ANNUAL) ** (1 / trading_days) - 1
    excess = port_daily - rf_daily
    sharpe = float((excess.mean() * trading_days) / ann_vol) if (ann_vol and ann_vol > 0) else float("nan")

    if aligned["bench"].var() != 0 and len(aligned) > 1:
        cov = np.cov(aligned["port"], aligned["bench"])[0, 1]
        var_bench = aligned["bench"].var()
        beta = float(cov / var_bench) if var_bench != 0 else float("nan")
    else:
        beta = float("nan")

    return Metrics(
        total_value=total_value,
        weights=weights.sort_values(ascending=False),
        daily_ret_series=port_daily,
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=sharpe,
        beta=beta
    )

def main():
    tickers = sorted(list(PORTFOLIO.keys()))
    if len(tickers) == 0:
        print("PORTFOLIO is empty. Please add tickers/shares in the config section.")
        sys.exit(1)

    # Make end inclusive-ish by nudging +1 day to avoid boundary issues
    start_dt = datetime.today() - timedelta(days=LOOKBACK_DAYS)
    end_dt = datetime.today() + timedelta(days=1)

    start = start_dt.strftime("%Y-%m-%d")
    end = end_dt.strftime("%Y-%m-%d")

    # Fetch portfolio + benchmark
    prices = fetch_prices(tickers, start, end)
    bench_df = fetch_prices([BENCHMARK], start, end)
    bench_prices = bench_df.iloc[:, 0]  # DF -> Series

    shares = pd.Series(PORTFOLIO, dtype=float).reindex(prices.columns).fillna(0.0)
    metrics = compute_metrics(prices, shares, bench_prices)

    # ----- Print summary -----
    print("\n=== PORTFOLIO SNAPSHOT ===")
    latest = prices.ffill().iloc[-1]
    pos_df = pd.DataFrame({
        "Shares": shares,
        "Last Price": latest,
        "Position Value": latest * shares,
        "Weight": metrics.weights.reindex(prices.columns).fillna(0.0)
    }).sort_values("Position Value", ascending=False)
    print(pos_df.round(2))
    print(f"\nTotal Portfolio Value: ${metrics.total_value:,.2f}")

    print("\n=== RISK / RETURN ===")
    print(f"Annualized Return: {metrics.ann_return:.2%}" if pd.notna(metrics.ann_return) else "Annualized Return: N/A")
    print(f"Annualized Volatility: {metrics.ann_vol:.2%}" if pd.notna(metrics.ann_vol) else "Annualized Volatility: N/A")
    print(f"Sharpe Ratio (rf={RISK_FREE_ANNUAL:.2%}): {metrics.sharpe:.2f}" if pd.notna(metrics.sharpe) else f"Sharpe Ratio (rf={RISK_FREE_ANNUAL:.2%}): N/A")
    print(f"Beta vs {BENCHMARK}: {metrics.beta:.2f}" if pd.notna(metrics.beta) else f"Beta vs {BENCHMARK}: N/A")

    # ----- Plots -----
    try:
        port_cum = (1 + metrics.daily_ret_series).cumprod()
        bench_cum = (1 + bench_prices.ffill().pct_change().dropna()).cumprod()
        aligned = pd.concat([port_cum, bench_cum], axis=1, join="inner")
        aligned.columns = ["Portfolio", "Benchmark"]

        aligned.plot(figsize=(10, 5), title="Cumulative Performance (Portfolio vs Benchmark)")
        plt.xlabel("Date")
        plt.ylabel("Growth of 1")
        plt.tight_layout()
        plt.savefig("performance.png", dpi=150)
        print("\nSaved plot: performance.png")

        (metrics.weights.sort_values(ascending=False) * 100).plot(
            kind="bar", figsize=(8, 4), title="Portfolio Weights (%)"
        )
        plt.ylabel("Percent")
        plt.tight_layout()
        plt.savefig("weights.png", dpi=150)
        print("Saved plot: weights.png")
    except Exception as e:
        print(f"\nPlotting skipped due to error: {e}")

if __name__ == "__main__":
    main()
