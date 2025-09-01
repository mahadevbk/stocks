import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import requests
import plotly.graph_objs as go

# Attempt to import forecasting libraries
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False

# Suppress pandas SettingWithCopyWarning
pd.options.mode.chained_assignment = None

# Streamlit app configuration
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📈 Dev's totally basic Stock Analyzer")

# Sidebar for user inputs
st.sidebar.header("Input Parameters")
symbol = st.sidebar.text_input("Stock Symbol (e.g., AAPL, TSLA)", value="AAPL").upper()
forecast_days = st.sidebar.slider("Forecast Days", min_value=30, max_value=365, value=180)
show_ma = st.sidebar.checkbox("Show Moving Averages", value=True)
show_rsi = st.sidebar.checkbox("Show RSI", value=True)
show_macd = st.sidebar.checkbox("Show MACD", value=True)
show_plots = st.sidebar.checkbox("Show Plots", value=True)

# NewsAPI key from Streamlit secrets or input
news_api_key = st.secrets.get("NEWS_API_KEY", None)
if not news_api_key:
    news_api_key = st.sidebar.text_input("NewsAPI Key (optional)", type="password")

# Main app logic
if st.button("Analyze Stock"):
    st.write(f"Analyzing {symbol}...")
    st.write("═"*40)

    try:
        stock = yf.Ticker(symbol)

        # --- Company Info ---
        st.subheader("Company Information")
        info = stock.info
        if info and info.get('longName'):
            info_to_show = {
                'Company Name': info.get('longName'),
                'Sector': info.get('sector'),
                'Industry': info.get('industry'),
                'Website': info.get('website'),
                'Market Cap': f"${info.get('marketCap', 0):,}",
                '52 Week High': info.get('fiftyTwoWeekHigh'),
                '52 Week Low': info.get('fiftyTwoWeekLow'),
                'Forward P/E': info.get('forwardPE')
            }
            for key, val in info_to_show.items():
                st.write(f"**{key}**: {val}")
        else:
            st.error(f"Could not retrieve company info for {symbol}. It might be an invalid ticker.")
            st.stop()

        # --- Historical Data ---
        st.subheader("Historical Data (5 Years)")
        hist = stock.history(period="5y")
        if hist.empty:
            st.error(f"No historical data found for {symbol}.")
            st.stop()

        st.write(f"**Latest Close Price**: ${hist['Close'].iloc[-1]:.2f}")

        if show_plots:
            fig = go.Figure(data=[go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name="Price"
            )])
            fig.update_layout(title=f"{symbol} Historical Price (5Y)")
            st.plotly_chart(fig, use_container_width=True)

        # --- Technical Indicators ---
        st.subheader("Technical Indicators")

        # Moving Averages
        if show_ma and show_plots:
            hist['MA20'] = hist['Close'].rolling(window=20).mean()
            hist['MA50'] = hist['Close'].rolling(window=50).mean()
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name='Close'))
            fig_ma.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], name='MA20'))
            fig_ma.add_trace(go.Scatter(x=hist.index, y=hist['MA50'], name='MA50'))
            fig_ma.update_layout(title=f"{symbol} Moving Averages (20 & 50 day)")
            st.plotly_chart(fig_ma, use_container_width=True)

        # RSI
        if show_rsi and show_plots:
            delta = hist['Close'].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.rolling(window=14, min_periods=14).mean()
            avg_loss = loss.rolling(window=14, min_periods=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=hist.index, y=rsi, name='RSI'))
            fig_rsi.add_hline(y=70, line=dict(dash="dash"), name="Overbought")
            fig_rsi.add_hline(y=30, line=dict(dash="dash"), name="Oversold")
            fig_rsi.update_layout(title=f"{symbol} Relative Strength Index (RSI)")
            st.plotly_chart(fig_rsi, use_container_width=True)

        # MACD
        if show_macd and show_plots:
            ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
            ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=hist.index, y=macd, name='MACD'))
            fig_macd.add_trace(go.Scatter(x=hist.index, y=signal, name='Signal'))
            fig_macd.update_layout(title=f"{symbol} MACD (12-26 EMA with 9-day Signal)")
            st.plotly_chart(fig_macd, use_container_width=True)

        # --- Forecasting ---
        st.subheader("Price Forecasts")

        # Prophet Forecast
        #if HAS_PROPHET:
        # Prophet Forecast
        if HAS_PROPHET:
            try:
                st.write("Running Prophet model...")
                forecast_data = hist.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
                # Remove timezone information from 'ds' column
                forecast_data['ds'] = forecast_data['ds'].dt.tz_localize(None)
                m = Prophet(daily_seasonality=True)
                m.fit(forecast_data)
                future = m.make_future_dataframe(periods=forecast_days)
                forecast = m.predict(future)

                st.write("**Prophet Forecast (Last 5 Days):**")
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5))

                if show_plots:
                    fig_forecast = go.Figure()
                    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted'))
                    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper Bound', line=dict(dash='dash', color='lightgray')))
                    fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower Bound', line=dict(dash='dash', color='lightgray')))
                    fig_forecast.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['y'], name='Historical'))
                    fig_forecast.update_layout(title=f"{symbol} Price Forecast (Prophet)")
                    st.plotly_chart(fig_forecast, use_container_width=True)
            except Exception as e:
                st.warning(f"Prophet forecasting could not be generated: {e}")
        else:
            st.info("Prophet is not installed. Skipping Prophet forecast.")

        # ARIMA Forecast
        if HAS_ARIMA:
            try:
                st.write("Running ARIMA model...")
                ts = hist['Close'].asfreq('B').ffill()
                model = ARIMA(ts, order=(5,1,0))
                res = model.fit()
                arima_forecast = res.get_forecast(steps=forecast_days)
                mean_forecast = arima_forecast.predicted_mean
                conf_int = arima_forecast.conf_int()

                st.write("**ARIMA (5,1,0) Forecast (Last 5 Days):**")
                st.dataframe(mean_forecast.tail(5))

                if show_plots:
                    fig_arima = go.Figure()
                    fig_arima.add_trace(go.Scatter(x=ts.index, y=ts, name='Historical'))
                    future_index = pd.date_range(ts.index[-1] + pd.offsets.BDay(), periods=forecast_days, freq='B')
                    fig_arima.add_trace(go.Scatter(x=future_index, y=mean_forecast, name='ARIMA Forecast'))
                    fig_arima.add_trace(go.Scatter(x=future_index, y=conf_int.iloc[:,0], name='Lower Bound', line=dict(dash='dash', color='lightgray')))
                    fig_arima.add_trace(go.Scatter(x=future_index, y=conf_int.iloc[:,1], name='Upper Bound', line=dict(dash='dash', color='lightgray')))
                    fig_arima.update_layout(title=f"{symbol} Price Forecast (ARIMA)")
                    st.plotly_chart(fig_arima, use_container_width=True)
            except Exception as e:
                st.warning(f"ARIMA forecasting could not be generated: {e}")
        else:
            st.info("statsmodels is not installed. Skipping ARIMA forecast.")

        # --- Analyst Price Targets ---
        st.subheader("Analyst Price Targets")
        target_mean = info.get("targetMeanPrice")
        if target_mean:
            st.write(f"**Mean Target Price**: ${info.get('targetMeanPrice')}")
            st.write(f"**High Target Price**: ${info.get('targetHighPrice')}")
            st.write(f"**Low Target Price**: ${info.get('targetLowPrice')}")
        else:
            st.write("No price target data available.")

        # --- Latest News ---
        if news_api_key:
            st.subheader("Latest News")
            try:
                url = f"https://newsapi.org/v2/everything?q={info.get('longName')}&apiKey={news_api_key}&language=en&sortBy=publishedAt&pageSize=5"
                response = requests.get(url)
                response.raise_for_status()
                articles = response.json().get("articles", [])
                if articles:
                    for art in articles:
                        st.write(f"**Title**: {art['title']}")
                        st.write(f"**Source**: {art['source']['name']}")
                        st.write(f"**Link**: [{art['url']}]({art['url']})")
                else:
                    st.write("No recent news found via NewsAPI.")
            except Exception as e:
                st.warning(f"Unable to fetch news: {e}")
        else:
            st.info("No NewsAPI key provided. Skipping news fetch.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
