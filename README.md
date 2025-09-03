﻿# Finance Portfolio Tracker

Interactive Streamlit app with Beginner & Advanced views. Uses free yfinance data.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://finance-portfolio-tracker.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<p align="center">
  <img src="docs/performance.png" width="65%" />
  <img src="docs/weights.png" width="30%" />
</p>

## ðŸš€ Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ðŸŒ± Beginner Mode

- **Portfolio Health Score** (diversification, ups & downs, market sensitivity)
- Donut charts (holdings & sectors)
- 1-day winners/losers heat table
- Goals & â€œwhat-if I invest moreâ€ slider
- Plain-English explanations of risk metrics

## ðŸ§  Advanced Mode

- **Return per unit of risk (Sharpe)**  
- **Moves vs market (Beta)**  
- **Annualized return/volatility**  
- **Max drawdown**  
- Cumulative performance vs benchmark

---

## ðŸ–¼ï¸ Screenshots

<p align="center">
  <img src="docs/screenshot1.png" width="800" /><br/>
  <img src="docs/screenshot2.png" width="800" /><br/>
  <img src="docs/screenshot3.png" width="800" />
</p>


## ðŸ“Š Example Inputs

Copy & paste into the app sidebar:

- `AAPL:50, MSFT:30, AMZN:10, GOOGL:15, TSLA:5`  
- `NVDA:10, JPM:25, JNJ:15, XOM:20, WMT:30`  
- `KO:40, PEP:30, MCD:10, JNJ:10, PG:10`

---

## ðŸ–¼ï¸ Demo

<p align="center">
  <img src="docs/demo.gif" width="800">
</p>

---

## ðŸ“‚ Project Structure

```
finance-portfolio-tracker/
â”œâ”€ app.py                 # Streamlit app (Beginner + Advanced modes)
â”œâ”€ portfolio_tracker.py   # CLI script (prints stats + saves charts)
â”œâ”€ requirements.txt       # Dependencies
â”œâ”€ README.md              # Project docs (this file)
â”œâ”€ LICENSE                # MIT License
â”œâ”€ docs/                  # Screenshots, demo gif
â””â”€ examples/              # Sample portfolio inputs
```

---

## ðŸ”§ Tech Stack

- [Streamlit](https://streamlit.io/) â€” interactive dashboards
- [yfinance](https://pypi.org/project/yfinance/) â€” free financial data
- [pandas](https://pandas.pydata.org/) & [numpy](https://numpy.org/) â€” data wrangling
- [matplotlib](https://matplotlib.org/) â€” charts

---

## ðŸ“œ License

MIT â€” see [LICENSE](LICENSE).

---

## ðŸ¤ Contributing

Pull requests welcome! Please open an issue first for discussion.  
For major changes, fork the repo and create a feature branch.

---

## â­ Acknowledgements

- Yahoo Finance data (via yfinance)
- Streamlit Community Cloud for free deployment
