import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
from groq import Groq
from search_engine import AISearch
aisearch = AISearch()

def load_stock_data(stock_data, intervals=['1d', '1h']):
    summaries = {}
    for interval in intervals:
        if stock_data is not None and not stock_data.empty:
            # Generate a summary for the last available data point
            last_summary = stock_data.iloc[-1].to_dict()
            summary_formatted = {f"{key}_{interval}": value for key, value in last_summary.items()}
            summaries[interval] = summary_formatted
        else:
            summaries[interval] = {'error': f'No data available for {interval} interval.'}
    return summaries

@st.cache_data(ttl=3600) 
def fetch_and_calculate_indicators(symbol, interval):
    # Adjust the start date based on the interval
    end_date = datetime.today()
    if interval == '1d':
        start_date = end_date - timedelta(days=120)  # 4 months for daily data
    else:
        # Adjust for hourly and 15-minute data to ensure within the 60-day limit
        start_date = end_date - timedelta(days=60)  # Adjusted for yfinance limitation
    
    try:
        # Fetch stock data using yfinance with the specified interval
        stock_data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval=interval)
        
        if stock_data.empty:
            st.error(f"No data returned for {symbol} using interval {interval}.")
            stock_data = None  # Exit the function
        
        # Calculate technical indicators using pandas-ta
        stock_data.ta.macd(append=True)
        stock_data.ta.rsi(append=True)
        stock_data.ta.bbands(append=True)
        stock_data.ta.obv(append=True)
        stock_data.ta.sma(length=20, append=True)
        stock_data.ta.ema(length=50, append=True)
        stock_data.ta.stoch(append=True)
        stock_data.ta.adx(append=True)
        stock_data.ta.willr(append=True)
        stock_data.ta.cmf(append=True)
        stock_data.ta.psar(append=True)
        
        # Convert OBV to million and handle MACD histogram naming
        stock_data['OBV_in_million'] = stock_data['OBV'] / 1e6
        stock_data['MACD_histogram_12_26_9'] = stock_data['MACDh_12_26_9']
        
    except Exception as e:
        print(f"Error fetching data for {symbol} at interval {interval}: {str(e)}")
        stock_data = None  # Ensure we return None to indicate failure

    return stock_data

import plotly.graph_objects as go
@st.cache_data(ttl=3600,show_spinner=False) 
def create_chart_for_indicator(stock_data, indicator_name, title, legend=True, hlines=None):
    """
    Creates an interactive figure for a specific stock indicator for displaying in Streamlit using Plotly.

    Parameters:
    - stock_data: DataFrame containing the stock data and indicators
    - indicator_name: list of column names in stock_data to plot
    - title: str, the title of the chart
    - legend: bool, whether to display the legend
    - hlines: list of tuples (y, color, dash_style) for horizontal lines

    Returns:
    - fig: The Plotly figure object to be displayed in Streamlit.
    """
    fig = go.Figure()

    # Plot each indicator as a separate line
    for name in indicator_name:
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data[name], mode='lines', name=name))

    # Add horizontal lines if specified
    if hlines:
        for hline in hlines:
            # Ensure the dash style is one of Plotly's accepted values
            # For a dashed line similar to Matplotlib's '--', use 'dash'
            dash_style = hline[2] if hline[2] in ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot'] else 'solid'
            
            fig.add_shape(type="line",
                          x0=stock_data.index.min(),
                          y0=hline[0],
                          x1=stock_data.index.max(),
                          y1=hline[0],
                          line=dict(color=hline[1], dash=dash_style),
                         )
            # Optionally add annotation next to the line
            fig.add_annotation(x=stock_data.index.mean(), y=hline[0], text=f"{hline[0]}", showarrow=False)

    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Value", legend_title="Indicator", height =400)
    fig.update_xaxes(tickangle=-45)
    return fig

@st.cache_data(ttl=3600,show_spinner=False) 
def create_separate_charts(stock_data):
    """
    Generates separate charts for key stock indicators and displays them in two columns in Streamlit.
    
    Parameters:
    - stock_data: DataFrame containing the stock data and indicators
    """
    # Define configurations for different charts
    charts_config = [
        {'indicator_name': ['Adj Close', 'EMA_50', 'SMA_20'], 'title': 'Price Trend'},
        {'indicator_name': ['OBV'], 'title': 'On-Balance Volume (OBV) Indicator'},
        {'indicator_name': ['MACD_12_26_9', 'MACDh_12_26_9'], 'title': 'MACD Indicator'},
        {'indicator_name': ['RSI_14'], 'title': 'RSI Indicator', 'hlines': [(70, 'red', '--'), (30, 'green', '--')]},
        {'indicator_name': ['BBU_5_2.0', 'BBM_5_2.0', 'BBL_5_2.0', 'Adj Close'], 'title': 'Bollinger Bands'},
        {'indicator_name': ['STOCHk_14_3_3', 'STOCHd_14_3_3'], 'title': 'Stochastic Oscillator'},
        {'indicator_name': ['WILLR_14'], 'title': 'Williams %R'},
        {'indicator_name': ['ADX_14'], 'title': 'Average Directional Index (ADX)'},
        {'indicator_name': ['CMF_20'], 'title': 'Chaikin Money Flow (CMF)'}
    ]
    
    # Generate and display each chart directly in the app
    for config in charts_config:
        fig = create_chart_for_indicator(
            stock_data=stock_data,
            indicator_name=config['indicator_name'],
            title=config['title'],
            legend=True,
            hlines=config.get('hlines', None)
        )
        st.plotly_chart(fig, use_container_width=True)

@st.cache_data(ttl=3600,show_spinner=False) 
def run_openai(timeframe,symbol,last_day_summary):
    st.session_state.ai_key = st.secrets["GROQ_API_KEY"]
      #st.session_state.openai_key = st.secrets["AI_KEY"]
    client = Groq(api_key =st.session_state.ai_key)
    

    latest_news = aisearch.serch_prompt_generate(symbol,search_mode=True)
    
    system_prompt = f"""
        Assume the role as a leading Technical Analysis (TA) expert in the stock/crypto market.
        Your mastery encompasses both stock and crypto fundamentals and intricate technical indicators.
        You possess the ability to decode complex market dynamics,
        providing clear insights and recommendations backed by a thorough understanding of interrelated factors.
        Your expertise extends to practical tools like the pandas_ta module,
        allowing you to navigate data intricacies with ease.
        Use this latest news as a reference in decision. {latest_news}.
        As a TA authority, your role is to decipher market trends, make informed predictions, and offer valuable perspectives.
        Answer the following in a simple term and markdown format: Stock /Crypto Name, Executive Summary,Technical Analysis,Investment Strategies,News Summary. Use Makrdown h3 Tags for titles.
        1. What this given stock symbol represents?
        2. Write overall executive summary for direction and predictive movement.
        2. Given TA data as below on the last trading {timeframe}, what will be the next few {timeframe} possible stock price movement?
        3. Give me idea as both long term investment and short term trading.
        5. Share summary of latest news on this ticker along with reference.

         
        """
    response = client.chat.completions.create (
        #model = "gpt-4-1106-preview",
        model = "llama-3.1-70b-versatile",
        #model = 'gpt-3.5-turbo',
        messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""The token is {symbol} ,\nHere is the summary indicators.\n{last_day_summary}"""}],
        max_tokens=1000
    )
    ai_response = response.choices[0].message.content
    #wrapped_reply = textwrap.fill(reply, width=100)

    return ai_response


def streamlit_app():
    st.set_page_config(layout="wide")
    # with st.sidebar:
    #     st.write("## Built By")
    #     st.write("Name: Nyein Chan Ko Ko")
    #     st.write("GitHub: (https://github.com/nchanko)")
    col1,col2,col3 = st.columns([1,3,1])
    with col1:
        st.image('stocklyzer.png',width=150)
    with col2:
        st.title("StockLyzer")
        st.text("Here you can view the stock data and make predictions")
        
    with col3:
        st.image("qr_code.png", width=100)
    st.markdown("**Some information on this page is AI-generated. This app is developed for educational purposes only and is not advisable to rely on it for financial decision-making.**")

    symbols = ['AAPL','ADBE','ADA-USD', 'AMD', 'ARB11841-USD', 'AVAX-USD', 'BNB-USD', 'BTC-USD', 'COIN', 'DOGE-USD', 'ETH-USD', 'GOOGL', 'ICP-USD', 'INTC', 'KSM-USD', 'LINK-USD', 'MATIC-USD', 'MSTR', 'MSFT', 'NEAR-USD', 'NVDA', 'SEI-USD', 'SOL-USD', 'TSLA', 'TSM','Custom symbol...']
    
    col1, col2, col3 = st.columns([4, 4, 2])
    with col1:
        with st.container():
            selected_symbol = st.selectbox("Select a Symbol", symbols)

            # Check if the user selected the option to enter a custom symbol
            if selected_symbol == 'Custom symbol...':
                custom_symbol = st.text_input("Enter Custom Symbol")

                # Use the custom symbol if provided, otherwise keep showing the input field
                if custom_symbol:
                    selected_symbol = custom_symbol

    with col2:
        interval_options = {'Daily': '1d', 'Hourly': '1h'}
        selected_interval = st.selectbox("Choose Interval", list(interval_options.keys()), format_func=lambda x: x)
        interval_value = interval_options[selected_interval]

    data_loaded = False
    with col3:
        st.text("")
        st.text("")
        if st.button('Predict', key='load_data'):
            data_loaded = True

    if data_loaded:
        with st.spinner(f'Loading data for {selected_symbol}...'):
            # Assuming fetch_and_calculate_indicators & load_stock_data definitions are provided here
            # Simulate fetching and calculating indicators
            # stock_data = fetch_and_calculate_indicators(selected_symbol, interval_value)
            # summaries = load_stock_data(selected_symbol, stock_data, intervals=[interval_value])
            # Simulated placeholders for stock_data and summaries
            stock_data = fetch_and_calculate_indicators(selected_symbol, interval_value)
            if stock_data is not None:
                summaries = load_stock_data(stock_data, intervals=[interval_value])

        textcol, chartcol = st.columns([4, 6])
        
        with textcol:
            if stock_data is not None:
                
                # Simulated AI response - replace with actual function call
                ai_response = run_openai(interval_value, selected_symbol, summaries)
                st.markdown(f"## \n\n{ai_response}")
                st.markdown("**This analysis has been generated using AI and is intended solely for educational purposes. It is not advisable to rely on it for financial decision-making.**")
                st.markdown(f"### **Summary for {interval_value} interval**")
            
                #st.json(summaries)

        with chartcol:
            if stock_data is not None:
                st.markdown(f"## **Stock Data for {selected_symbol}**")
                # Create separate charts function call should be placed here
                create_separate_charts(stock_data)


    st.write("---")

    # Buy Me a Coffee Button
    st.markdown(
        """
        ## Enjoying this app?
        If you find this app helpful, consider buying me a coffee! ☕️

        <a href="https://www.buymeacoffee.com/nyeinchankoko" target="_blank">
            <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" 
                 alt="Buy Me A Coffee" 
                 style="height: 60px !important; width: 217px !important;">
        </a>
        """,
        unsafe_allow_html=True
    )

    st.write("#### About This App")
    st.write("""This app demonstrates how to analyze custom CSV data for currency exchange rates, calculate technical indicators, and display the data and indicators using Plotly in Streamlit.
                 \nNyein Chan Ko Ko [nchanko](https://github.com/nchanko)""")

if __name__ == '__main__':
    streamlit_app()