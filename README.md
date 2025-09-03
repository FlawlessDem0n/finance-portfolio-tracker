# Finance Portfolio Tracker

Interactive Streamlit app with Beginner & Advanced views. Uses free yfinance data.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<p align="center">
  <img src="docs/performance.png" width="65%" />
  <img src="docs/weights.png" width="30%" />
</p>

## 🚀 Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌱 Beginner Mode

- **Portfolio Health Score** (diversification, ups & downs, market sensitivity)
- Donut charts (holdings & sectors)
- 1-day winners/losers heat table
- Goals & “what-if I invest more” slider
- Plain-English explanations of risk metrics

## 🧠 Advanced Mode

- **Return per unit of risk (Sharpe)**  
- **Moves vs market (Beta)**  
- **Annualized return/volatility**  
- **Max drawdown**  
- Cumulative performance vs benchmark

---

## 🖼️ Screenshots

<p align="center">
  <img src="docs/screenshot1.png" width="800" /><br/>
  <img src="docs/screenshot2.png" width="800" /><br/>
  <img src="docs/screenshot3.png" width="800" />
</p>


## 📊 Example Inputs

Copy & paste into the app sidebar:

- `AAPL:50, MSFT:30, AMZN:10, GOOGL:15, TSLA:5`  
- `NVDA:10, JPM:25, JNJ:15, XOM:20, WMT:30`  
- `KO:40, PEP:30, MCD:10, JNJ:10, PG:10`

---

## 🖼️ Demo

<p align="center">
  <img src="docs/demo.gif" width="800">
</p>

---

## 📂 Project Structure

```
finance-portfolio-tracker/
├─ app.py                 # Streamlit app (Beginner + Advanced modes)
├─ portfolio_tracker.py   # CLI script (prints stats + saves charts)
├─ requirements.txt       # Dependencies
├─ README.md              # Project docs (this file)
├─ LICENSE                # MIT License
├─ docs/                  # Screenshots, demo gif
└─ examples/              # Sample portfolio inputs
```

---

## 🔧 Tech Stack

- [Streamlit](https://streamlit.io/) — interactive dashboards
- [yfinance](https://pypi.org/project/yfinance/) — free financial data
- [pandas](https://pandas.pydata.org/) & [numpy](https://numpy.org/) — data wrangling
- [matplotlib](https://matplotlib.org/) — charts

---

## 📜 License

MIT — see [LICENSE](LICENSE).

---

## 🤝 Contributing

Pull requests welcome! Please open an issue first for discussion.  
For major changes, fork the repo and create a feature branch.

---

## ⭐ Acknowledgements

- Yahoo Finance data (via yfinance)
- Streamlit Community Cloud for free deployment
