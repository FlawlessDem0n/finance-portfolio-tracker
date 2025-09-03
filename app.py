import math
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------- PAGE / HEADER -----------------------
st.set_page_config(page_title="Portfolio Tracker", layout="wide")
st.title("📊 Portfolio Tracker (yfinance)")
st.caption("Beginner & Advanced views. Free data via yfinance (no API key).")

# ----------------------- INPUTS -----------------------
default_portfolio = "AAPL:50, MSFT:20, NVDA:10, JPM:25, WMT:30"
inp = st.sidebar.text_input("Portfolio (TICKER:SHARES, comma-separated)", value=default_portfolio)
benchmark = st.sidebar.text_input("Benchmark ticker", value="^GSPC")
lookback_days = st.sidebar.slider("Lookback (days)", 30, 1095, 365)
risk_free_annual = st.sidebar.number_input("Risk-free rate (annual, e.g., 0.02 = 2%)", value=0.00, step=0.01, format="%.4f")
st.sidebar.markdown("---")
st.sidebar.caption("Tip: Try a short lookback (90–180) or longer (365–730) to see different trends.")

def parse_portfolio(s: str) -> Dict[str, float]:
    items = [x.strip() for x in s.split(",") if x.strip()]
    out: Dict[str, float] = {}
    for item in items:
        try:
            t, sh = item.split(":")
            t = t.strip().upper()
            out[t] = out.get(t, 0.0) + float(sh.strip())
        except Exception:
            # skip malformed parts
            pass
    return out

portfolio = parse_portfolio(inp)
if not portfolio:
    st.warning("Please enter a valid portfolio like `AAPL:50, MSFT:20`.")
    st.stop()

# Make end inclusive-ish to avoid boundary gaps/holidays
start_dt = datetime.today() - timedelta(days=lookback_days)
end_dt = datetime.today() + timedelta(days=1)
start = start_dt.strftime("%Y-%m-%d")
end = end_dt.strftime("%Y-%m-%d")

# ----------------------- DATA LAYER -----------------------
@st.cache_data(show_spinner=False)
def fetch_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetch adjusted close prices using yfinance.
    - auto_adjust=True => 'Close' already adjusted for splits/dividends.
    - forward/backward fill to avoid NaN wipeouts on short windows/holidays.
    """
    if not isinstance(tickers, (list, tuple)):
        tickers = [tickers]

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        group_by="column",
    )

    if raw is None or raw.empty:
        raise ValueError(f"No price data for {tickers} between {start} and {end}.")

    # Multi vs single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise KeyError("Downloaded data has no 'Close' field.")
        close = raw["Close"].copy()
        # Keep only requested tickers, in order
        close = close.loc[:, [t for t in tickers if t in close.columns]]
    else:
        if "Close" not in raw.columns:
            raise KeyError("Downloaded data has no 'Close' field.")
        close = raw[["Close"]].copy()
        name = tickers[0] if len(tickers) >= 1 else "TICKER"
        close.columns = [name]

    # Robustness around weekends/holidays
    close = close.ffill().bfill()
    if close.isna().all().all():
        raise ValueError(f"Fetched only NaN values for {tickers}. Try a different lookback/tickers.")
    # Drop columns still all-NaN (very rare)
    close = close.loc[:, ~close.isna().all(axis=0)]
    if close.empty:
        raise ValueError("All fetched prices are NaN. Try a longer lookback or different tickers.")
    return close

@st.cache_data(show_spinner=False)
def get_sectors(tickers: List[str]) -> Dict[str, str]:
    """
    Best-effort sector mapping via yfinance.
    Falls back to 'Unknown' if not available quickly.
    Cached to reduce API calls.
    """
    sectors: Dict[str, str] = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).get_info()  # yfinance may warn; still works for sector
            sector = info.get("sector") or info.get("industry") or "Unknown"
            sectors[t] = sector
        except Exception:
            sectors[t] = "Unknown"
    return sectors

def max_drawdown(series: pd.Series) -> float:
    """Maximum drawdown of cumulative return series (0..1)."""
    if series.empty:
        return np.nan
    running_max = series.cummax()
    dd = (series - running_max) / running_max
    return float(dd.min())

def annualize_return(daily_returns: pd.Series, trading_days: int = 252) -> float:
    if len(daily_returns) == 0:
        return np.nan
    cum = (1 + daily_returns).prod()
    years = len(daily_returns) / trading_days
    return float(cum ** (1 / years) - 1) if years > 0 else np.nan

def compute_metrics(prices: pd.DataFrame,
                    shares: pd.Series,
                    bench_prices: pd.Series,
                    rf_annual: float) -> Dict[str, float]:
    prices = prices.ffill().bfill()
    bench_prices = bench_prices.ffill().bfill()

    latest = prices.iloc[-1]
    posval = latest * shares
    total_value = float(posval.sum())
    weights = (posval / total_value).fillna(0.0) if total_value > 0 else shares * 0

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
    ann_vol = float(port_daily.std() * math.sqrt(trading_days)) if len(port_daily) > 1 else np.nan
    ann_ret = annualize_return(port_daily, trading_days)

    rf_daily = (1 + rf_annual) ** (1 / trading_days) - 1
    excess = port_daily - rf_daily
    sharpe = float((excess.mean() * trading_days) / ann_vol) if (ann_vol and ann_vol > 0) else np.nan

    beta = (np.cov(aligned["port"], aligned["bench"])[0, 1] / aligned["bench"].var()
            if aligned["bench"].var() != 0 and len(aligned) > 1 else np.nan)

    port_cum = (1 + port_daily).cumprod()
    mdd = max_drawdown(port_cum)  # negative number (e.g., -0.32)

    # One-day winners/losers
    one_day_change = prices.ffill().bfill().pct_change().iloc[-1].reindex(latest.index)

    return {
        "total_value": total_value,
        "weights": weights,
        "ann_vol": ann_vol,
        "ann_ret": ann_ret,
        "sharpe": sharpe,
        "beta": float(beta) if beta is not np.nan else np.nan,
        "mdd": mdd,
        "one_day_change": one_day_change,
        "port_daily": port_daily,
        "bench_daily": bench_daily,
        "latest_prices": latest,
        "posval": posval,
        "port_cum": port_cum,
    }

def diversification_score(weights: pd.Series) -> Tuple[float, float]:
    """
    Herfindahl-Hirschman Index (HHI) to score diversification.
    Score 0..100: 100 is max diversified (equal weights), 0 is concentrated (100% single asset).
    """
    w = weights.fillna(0.0).values
    if w.sum() == 0:
        return 0.0, 1.0
    hhi = float((w ** 2).sum())
    n = (w > 0).sum()
    if n <= 1:
        return 0.0, hhi
    # Normalize HHI to 0..1, then map to 0..100
    # best = 1/n ; worst = 1
    norm = (hhi - 1 / n) / (1 - 1 / n)
    score = (1 - norm) * 100
    return max(0.0, min(100.0, score)), hhi

def volatility_score(ann_vol: float) -> float:
    """
    Map annualized vol to 0..100 (lower vol => higher score).
    10% vol ~ 100, 40% vol ~ 0 (linear clamp).
    """
    if np.isnan(ann_vol):
        return np.nan
    lo, hi = 0.10, 0.40
    if ann_vol <= lo:
        return 100.0
    if ann_vol >= hi:
        return 0.0
    return float(100 * (1 - (ann_vol - lo) / (hi - lo)))

def beta_score(beta: float) -> float:
    """
    Penalize beta > 1 (riskier than market). Beta==1 => 80, Beta 0.8=> ~96, Beta 1.5 => ~60.
    """
    if np.isnan(beta):
        return np.nan
    base = 100 - max(0.0, (beta - 1.0)) * 40  # 0.5 beta above 1.0 => -20 points
    return max(0.0, min(100.0, base))

def health_score(weights: pd.Series, ann_vol: float, beta: float) -> Tuple[float, Dict[str, float]]:
    """
    Weighted blend of diversification, volatility, and beta.
    """
    div_s, hhi = diversification_score(weights)
    vol_s = volatility_score(ann_vol)
    bet_s = beta_score(beta)
    # Weights: Diversification 0.4, Vol 0.3, Beta 0.3
    parts = {
        "Diversification": div_s,
        "Volatility": vol_s,
        "Market Sensitivity": bet_s
    }
    vals = [v for v in parts.values() if not np.isnan(v)]
    if not vals:
        return np.nan, parts
    score = 0.4 * (div_s if not np.isnan(div_s) else 0) + \
            0.3 * (vol_s if not np.isnan(vol_s) else 0) + \
            0.3 * (bet_s if not np.isnan(bet_s) else 0)
    return float(score), parts

# ----------------------- FETCH DATA -----------------------
try:
    tickers = list(portfolio.keys())
    prices = fetch_prices(tickers, start, end)
    bench_df = fetch_prices([benchmark], start, end)
    bench_prices = bench_df.iloc[:, 0]
except Exception as e:
    st.error(f"Data fetch failed: {e}")
    st.stop()

latest = prices.ffill().bfill().iloc[-1]
shares = pd.Series(portfolio).reindex(prices.columns).astype(float).fillna(0.0)

# ----------------------- METRICS -----------------------
metrics = compute_metrics(prices, shares, bench_prices, risk_free_annual)

# ----------------------- SECTORS -----------------------
sectors = get_sectors(list(prices.columns))
sector_series = pd.Series({t: sectors.get(t, "Unknown") for t in prices.columns})
sector_weights = metrics["weights"].groupby(sector_series).sum().sort_values(ascending=False)

# ----------------------- TABS -----------------------
tab1, tab2 = st.tabs(["🌱 Beginner Mode", "🧠 Advanced Mode"])

# ========== BEGINNER MODE ==========
with tab1:
    colA, colB = st.columns([1, 1])
    with colA:
        st.subheader("Portfolio Snapshot")
        df_hold = pd.DataFrame({
            "Shares": shares,
            "Last Price": metrics["latest_prices"],
            "Position Value": metrics["posval"],
            "Weight": metrics["weights"]
        }).sort_values("Position Value", ascending=False)
        st.dataframe(
            df_hold.style.format({
                "Last Price": "${:,.2f}",
                "Position Value": "${:,.2f}",
                "Weight": "{:.2%}"
            })
        )
        st.metric("Total Portfolio Value", f"${metrics['total_value']:,.2f}")

    with colB:
        st.subheader("Who’s up / down today?")
        movers = pd.DataFrame({
            "1D Change (%)": (metrics["one_day_change"] * 100.0).round(2),
            "Weight": metrics["weights"]
        }).sort_values("1D Change (%)", ascending=False)
        styled = movers.style.format({"1D Change (%)": "{:+.2f}%", "Weight": "{:.2%}"}).background_gradient(
            subset=["1D Change (%)"], cmap="RdYlGn"
        )
        st.dataframe(styled, use_container_width=True)

    # Health score + explanation
    st.subheader("Portfolio Health Score")
    score, parts = health_score(metrics["weights"], metrics["ann_vol"], metrics["beta"])
    cols = st.columns([1, 2])

    with cols[0]:
        # Simple gauge-like text
        if not np.isnan(score):
            st.metric("Overall Score (0–100)", f"{score:.0f}")
        else:
            st.write("Insufficient data to score.")
        # Explain parts in plain English
        st.markdown("**What this means (plain English):**")
        st.markdown(
            "- **Diversification** = Spread across multiple stocks/sectors (more spread → better).\n"
            "- **Ups & Downs (Volatility)** = How bumpy the ride is each year.\n"
            "- **Moves vs Market (Beta)** = 1.0 ≈ market-like. Higher than 1 moves more than market."
        )

    with cols[1]:
        parts_df = pd.DataFrame.from_dict(parts, orient="index", columns=["Score"]).sort_values("Score", ascending=False)
        st.bar_chart(parts_df)

    # Donut charts (Holdings + Sector)
    colC, colD = st.columns([1, 1])

    def donut_chart(labels: List[str], values: List[float], title: str):
        fig, ax = plt.subplots(figsize=(5, 5))
        wedges, _ = ax.pie(values, wedgeprops=dict(width=0.45), startangle=90)
        ax.set(aspect="equal", title=title)
        # Add tiny white circle
        centre_circle = plt.Circle((0, 0), 0.70, fc="white")
        fig.gca().add_artist(centre_circle)
        ax.legend(labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
        st.pyplot(fig, clear_figure=True)

    with colC:
        st.subheader("Holdings Mix")
        w = metrics["weights"].sort_values(ascending=False)
        donut_chart(w.index.tolist(), (w.values * 100).tolist(), "Portfolio Weights (%)")

    with colD:
        st.subheader("Sector Mix")
        if not sector_weights.empty:
            donut_chart(sector_weights.index.tolist(), (sector_weights.values * 100).tolist(), "By Sector (%)")
        else:
            st.info("Sector data not available.")

    # Goals & What-if section
    st.subheader("Goals & What-if")
    gc1, gc2, gc3 = st.columns([1, 1, 1])
    with gc1:
        goal_amount = st.number_input("Target portfolio value ($)", value=float(round(metrics["total_value"] * 1.25, 2)))
    with gc2:
        goal_date = st.date_input("Target date", value=date.today().replace(year=date.today().year + 1))
    with gc3:
        extra_invest = st.slider("What if I invest an extra … ($)", min_value=0, max_value=100000, value=1000, step=500)

    # Goal progress
    days_to_goal = max(1, (goal_date - date.today()).days)
    needed_delta = max(0.0, goal_amount - metrics["total_value"])
    monthly_needed = needed_delta / (days_to_goal / 30.0)
    st.write(
        f"**Goal progress:** You are at **${metrics['total_value']:,.0f}** out of **${goal_amount:,.0f}**. "
        f"To reach this by **{goal_date}**, you’d need to contribute roughly **${monthly_needed:,.0f}/month** "
        f"(ignoring market returns)."
    )

    # What-if distribution (pro-rata by current weights; if no weights, equal split)
    w = metrics["weights"].copy()
    if w.sum() == 0 or w.isna().all():
        w = pd.Series(1 / len(shares), index=shares.index)
    add_allocation = (w * extra_invest).round(2)
    new_posval = metrics["posval"] + add_allocation
    new_total = float(new_posval.sum())
    new_weights = (new_posval / new_total).fillna(0.0)
    what_df = pd.DataFrame({
        "Add ($)": add_allocation,
        "New Weight": new_weights
    }).sort_values("Add ($)", ascending=False)
    st.write(f"**After investing ${extra_invest:,.0f}:** new total ≈ **${new_total:,.2f}**")
    st.dataframe(what_df.style.format({"Add ($)": "${:,.2f}", "New Weight": "{:.2%}"}))

    # Friendly warnings
    st.subheader("Quick Health Checks")
    warnings = []
    if metrics["weights"].max() > 0.5:
        warnings.append("Your portfolio is **highly concentrated** (one holding > 50%). Consider diversifying.")
    if not sector_weights.empty and sector_weights.iloc[0] > 0.5:
        warnings.append(f"You're heavily tilted to **{sector_weights.index[0]}** (> 50%).")
    if metrics["beta"] and not np.isnan(metrics["beta"]) and metrics["beta"] > 1.5:
        warnings.append("Your portfolio **moves much more than the market** (beta > 1.5). Expect bigger swings.")
    if metrics["ann_vol"] and not np.isnan(metrics["ann_vol"]) and metrics["ann_vol"] > 0.35:
        warnings.append("Your **ups & downs (volatility)** are quite high (> 35%/yr).")

    if warnings:
        for wmsg in warnings:
            st.warning(wmsg)
    else:
        st.success("All good! Your portfolio looks reasonably balanced for a beginner.")

    # Tiny risk tolerance “quiz”
    st.subheader("Risk Comfort Check (1-minute)")
    choice = st.radio(
        "How do you feel about temporary drops?",
        ["I prefer small, steady growth", "I can handle moderate ups & downs", "I want high growth and accept big drops"],
        index=1,
        horizontal=False,
    )
    if choice == "I prefer small, steady growth":
        st.info("Aim for **lower volatility** and **beta near 1 or below**. Consider increasing diversification.")
    elif choice == "I can handle moderate ups & downs":
        st.info("A **balanced** beta (~1.0–1.2) and good diversification is usually fine.")
    else:
        st.info("Higher beta/volatility is okay for you — but be ready for larger swings. Diversify to manage risk.")

# ========== ADVANCED MODE ==========
with tab2:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Risk & Return (Advanced)")
        st.write("**Return per Unit of Risk (Sharpe):** "
                 f"{metrics['sharpe']:.2f} _(rf={risk_free_annual:.2%})_" if pd.notna(metrics["sharpe"]) else "Sharpe: N/A")
        st.write(f"**Moves vs Market (Beta):** {metrics['beta']:.2f}" if pd.notna(metrics["beta"]) else "Beta: N/A")
        st.write(f"**Annualized Return:** {metrics['ann_ret']:.2%}" if pd.notna(metrics["ann_ret"]) else "Annualized Return: N/A")
        st.write(f"**Annualized Volatility:** {metrics['ann_vol']:.2%}" if pd.notna(metrics["ann_vol"]) else "Annualized Volatility: N/A")
        st.write(f"**Max Drawdown:** {metrics['mdd']:.2%}" if pd.notna(metrics["mdd"]) else "Max Drawdown: N/A")

    with col2:
        st.subheader("Cumulative Performance vs Benchmark")
        perf = pd.concat([
            metrics["port_cum"].rename("Portfolio"),
            (1 + metrics["bench_daily"]).cumprod().rename("Benchmark"),
        ], axis=1).dropna()
        st.line_chart(perf)

    st.subheader("Holdings (Full)")
    full_df = pd.DataFrame({
        "Shares": shares,
        "Last Price": metrics["latest_prices"],
        "Position Value": metrics["posval"],
        "Weight": metrics["weights"],
        "1D Change (%)": metrics["one_day_change"] * 100.0,
        "Sector": sector_series,
    }).sort_values("Position Value", ascending=False)
    st.dataframe(full_df.style.format({
        "Last Price": "${:,.2f}",
        "Position Value": "${:,.2f}",
        "Weight": "{:.2%}",
        "1D Change (%)": "{:+.2f}%"
    }))

    st.subheader("Sector Weights")
    if not sector_weights.empty:
        st.bar_chart((sector_weights * 100).to_frame("Weight %"))
    else:
        st.info("Sector data not available.")

# Footer tip
st.caption("Pro tip: Deploy on Streamlit Community Cloud and add a polished README with screenshots and plain-English explanations of each metric.")
