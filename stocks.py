import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import requests
import plotly.graph_objs as go

# Attempt to import forecasting libs
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except Exception:
    HAS_ARIMA = False

# ---------- CONFIG ----------
NEWS_API_KEY = "your_newsapi_key_here"  # <-- Replace with your NewsAPI key

# ---------- APP START ----------
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📈 Stock Analyzer with Sources")

# Sidebar controls
st.sidebar.header("Controls")
forecast_days = st.sidebar.number_input("Forecast horizon (days)", min_value=30, max_value=365, value=180, step=15)
show_ma = st.sidebar.checkbox("Show 20/50 MA", value=True)
show_rsi = st.sidebar.checkbox("Show RSI", value=True)
show_macd = st.sidebar.checkbox("Show MACD", value=True)

# User input
symbol = st.text_input("Enter Stock Symbol (e.g. AAPL, TSLA, MSFT):", "AAPL").upper()

if symbol:
    try:
        stock = yf.Ticker(symbol)

        # --- Company Info ---
        st.header("Company Information")
        info = stock.info
        if info:
            company_df = pd.DataFrame.from_dict(info, orient='index', columns=['Value'])
            st.write(company_df)
            st.caption("Source: Yahoo Finance (via yfinance)")

        # --- Historical Data ---
        st.header("Historical Stock Prices")
        hist = stock.history(period="5y")
        if not hist.empty:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name="Price"
            ))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Source: Yahoo Finance (via yfinance)")

        # --- Technical Indicators ---
        st.header("Technical Indicators")
        if not hist.empty:
            # Moving Averages
            if show_ma:
                hist['MA20'] = hist['Close'].rolling(window=20).mean()
                hist['MA50'] = hist['Close'].rolling(window=50).mean()
                fig_ma = go.Figure()
                fig_ma.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Close'))
                fig_ma.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], name='MA20'))
                fig_ma.add_trace(go.Scatter(x=hist.index, y=hist['MA50'], name='MA50'))
                st.subheader("Moving Averages (20 & 50 day)")
                st.plotly_chart(fig_ma, use_container_width=True)

            # RSI
            if show_rsi:
                delta = hist['Close'].diff()
                gain = delta.where(delta > 0, 0.0)
                loss = -delta.where(delta < 0, 0.0)
                avg_gain = gain.rolling(window=14, min_periods=14).mean()
                avg_loss = loss.rolling(window=14, min_periods=14).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=hist.index, y=rsi, name='RSI'))
                fig_rsi.add_hline(y=70, line=dict(dash="dash"))
                fig_rsi.add_hline(y=30, line=dict(dash="dash"))
                st.subheader("Relative Strength Index (RSI)")
                st.plotly_chart(fig_rsi, use_container_width=True)

            # MACD
            if show_macd:
                ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
                ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9, adjust=False).mean()
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=hist.index, y=macd, name='MACD'))
                fig_macd.add_trace(go.Scatter(x=hist.index, y=signal, name='Signal'))
                st.subheader("MACD (12-26 EMA with 9-day Signal)")
                st.plotly_chart(fig_macd, use_container_width=True)

            st.caption("Source: Calculated from Yahoo Finance historical price data")

        # --- Forecasting with Prophet ---
        st.header("Price Forecasts")
        if not hist.empty:
            forecast_tab1, forecast_tab2 = st.tabs(["Prophet", "ARIMA"])

            with forecast_tab1:
                if HAS_PROPHET:
                    try:
                        forecast_data = hist.reset_index()[['Date', 'Close']]
                        forecast_data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
                        m = Prophet(daily_seasonality=True)
                        m.fit(forecast_data)
                        future = m.make_future_dataframe(periods=int(forecast_days))
                        forecast = m.predict(future)

                        fig_forecast = go.Figure()
                        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted'))
                        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper Bound', line=dict(dash='dash')))
                        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower Bound', line=dict(dash='dash')))
                        fig_forecast.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['y'], name='Historical'))
                        st.plotly_chart(fig_forecast, use_container_width=True)
                        st.caption("Source: Prophet forecasting model (PyPI 'prophet') based on Yahoo Finance historical price data. Forecasts are statistical projections, not financial advice.")
                    except Exception as e:
                        st.warning(f"Prophet forecasting could not be generated: {e}")
                else:
                    st.info("Prophet is not installed in this environment. Install with: pip install prophet")

            # --- ARIMA Forecast ---
            with forecast_tab2:
                if HAS_ARIMA:
                    try:
                        # Use business-day frequency and forward-fill
                        ts = hist['Close'].asfreq('B').ffill()

                        # Small grid search for a low-cost model selection
                        candidate_orders = [(1,1,0), (1,1,1), (2,1,1), (5,1,0)]
                        best_model = None
                        best_aic = np.inf
                        best_order = None
                        for order in candidate_orders:
                            try:
                                model = ARIMA(ts, order=order)
                                res = model.fit()
                                if res.aic < best_aic:
                                    best_aic = res.aic
                                    best_model = res
                                    best_order = order
                            except Exception:
                                continue
                        if best_model is None:
                            raise RuntimeError("No ARIMA model could be fit.")

                        arima_forecast = best_model.get_forecast(steps=int(forecast_days))
                        mean_forecast = arima_forecast.predicted_mean
                        conf_int = arima_forecast.conf_int()

                        # Build plot
                        fig_arima = go.Figure()
                        fig_arima.add_trace(go.Scatter(x=ts.index, y=ts, name='Historical'))
                        future_index = pd.date_range(ts.index[-1] + pd.offsets.BDay(), periods=int(forecast_days), freq='B')
                        fig_arima.add_trace(go.Scatter(x=future_index, y=mean_forecast, name=f'ARIMA Forecast {best_order}'))
                        fig_arima.add_trace(go.Scatter(x=future_index, y=conf_int.iloc[:,0], name='Lower Bound', line=dict(dash='dash')))
                        fig_arima.add_trace(go.Scatter(x=future_index, y=conf_int.iloc[:,1], name='Upper Bound', line=dict(dash='dash')))
                        st.plotly_chart(fig_arima, use_container_width=True)
                        st.caption("Source: statsmodels ARIMA (PyPI 'statsmodels') based on Yahoo Finance historical price data. Forecasts are statistical projections, not financial advice.")
                    except Exception as e:
                        st.warning(f"ARIMA forecasting could not be generated: {e}")
                else:
                    st.info("statsmodels is not installed in this environment. Install with: pip install statsmodels")

        # --- Financials ---
        st.header("Financial Statements")
        st.subheader("Balance Sheet")
        st.write(stock.balance_sheet)
        st.subheader("Income Statement")
        st.write(stock.financials)
        st.subheader("Cash Flow")
        st.write(stock.cashflow)
        st.caption("Source: Yahoo Finance (via yfinance)")

        # --- Analyst Recommendations ---
        st.header("Analyst Recommendations")
        recs = stock.recommendations
        if recs is not None:
            st.write(recs.tail(20))
            st.caption("Source: Yahoo Finance (via yfinance)")

        # --- Analyst Price Targets ---
        st.header("Analyst Price Targets")
        try:
            target_mean = info.get("targetMeanPrice", None)
            target_high = info.get("targetHighPrice", None)
            target_low = info.get("targetLowPrice", None)
            if target_mean:
                st.metric("Mean Target Price", f"${target_mean}")
                st.metric("High Target Price", f"${target_high}")
                st.metric("Low Target Price", f"${target_low}")
            st.caption("Source: Yahoo Finance (via yfinance)")
        except Exception:
            st.warning("No price target data available.")

        # --- Latest News ---
        st.header("Latest News")
        try:
            url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}&language=en&sortBy=publishedAt&pageSize=5"
            response = requests.get(url)
            articles = response.json().get("articles", [])
            if articles:
                for art in articles:
                    st.subheader(art['title'])
                    st.write(art['description'])
                    st.markdown(f"[Read more]({art['url']})")
                    st.caption(f"Source: {art['source']['name']} (via NewsAPI)")
            else:
                st.info("No recent news found.")
        except Exception:
            st.warning("Unable to fetch news.")

    except Exception as e:
        st.error(f"Could not fetch data for {symbol}: {e}")

