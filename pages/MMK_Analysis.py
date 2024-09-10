import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from groq import Groq
from search_engine import AISearch

aisearch = AISearch()

# Constants for CSV file locations
CSV_FILE_PATH_USDMMK = st.secrets["MMK_USD_PATH"]
CSV_FILE_PATH_USDTMMK = st.secrets["USDT_CSV_PATH"]

@st.cache_data(ttl=3600) 
def load_csv_data(currency_pair):
    if currency_pair == 'USDMMK':
        df = pd.read_csv(CSV_FILE_PATH_USDMMK, parse_dates=['DATE'])
        df.set_index('DATE', inplace=True)
    elif currency_pair == 'USDTMMK':
        df = pd.read_csv(CSV_FILE_PATH_USDTMMK, parse_dates=['date'])
        df.set_index('date', inplace=True)
        # For USDTMMK, rename columns to match processing function
        df.rename(columns={'rate': 'avg USD_BUY'}, inplace=True)
        df['avg USD_SELL'] = df['avg USD_BUY']  # Dummy column for consistency
    return df

def calculate_indicators(df):
    if 'avg USD_SELL' in df.columns:
        # Calculate indicators only if relevant columns are present
        df['SMA_BUY_20'] = df['avg USD_BUY'].rolling(window=20).mean()
        df['EMA_BUY_50'] = df['avg USD_BUY'].ewm(span=50, adjust=False).mean()
        df['SMA_SELL_20'] = df['avg USD_SELL'].rolling(window=20).mean()
        df['EMA_SELL_50'] = df['avg USD_SELL'].ewm(span=50, adjust=False).mean()
        
        # Calculate Bollinger Bands for BUY
        df['BB_BUY_MIDDLE'] = df['avg USD_BUY'].rolling(window=20).mean()
        df['BB_BUY_UPPER'] = df['BB_BUY_MIDDLE'] + 2 * df['avg USD_BUY'].rolling(window=20).std()
        df['BB_BUY_LOWER'] = df['BB_BUY_MIDDLE'] - 2 * df['avg USD_BUY'].rolling(window=20).std()
        
        # Calculate spread
        df['SPREAD'] = df['avg USD_SELL'] - df['avg USD_BUY']
        
        # Calculate rate of change
        df['ROC_BUY'] = df['avg USD_BUY'].pct_change(periods=1) * 100
        df['ROC_SELL'] = df['avg USD_SELL'].pct_change(periods=1) * 100

        # Calculate RSI
        delta = df['avg USD_BUY'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def create_chart_for_indicator(df, indicator_names, title, legend=True, hlines=None):
    fig = go.Figure()
    
    for name in indicator_names:
        fig.add_trace(go.Scatter(x=df.index, y=df[name], mode='lines', name=name))
    
    if hlines:
        for hline in hlines:
            dash_style = hline[2] if hline[2] in ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot'] else 'solid'
            fig.add_shape(type="line",
                          x0=df.index.min(),
                          y0=hline[0],
                          x1=df.index.max(),
                          y1=hline[0],
                          line=dict(color=hline[1], dash=dash_style))
            fig.add_annotation(x=df.index.mean(), y=hline[0], text=f"{hline[0]}", showarrow=False)
    
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Value", legend_title="Indicator", height=400)
    fig.update_xaxes(tickangle=-45)
    return fig

@st.cache_data(ttl=3600, show_spinner=False)
def create_rsi_chart(df):
    fig = go.Figure()
    
    # Plot RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
    
    # Add horizontal lines for RSI thresholds
    fig.add_shape(type="line",
                  x0=df.index.min(),
                  y0=70,
                  x1=df.index.max(),
                  y1=70,
                  line=dict(color='red', dash='dash'))
    fig.add_shape(type="line",
                  x0=df.index.min(),
                  y0=30,
                  x1=df.index.max(),
                  y1=30,
                  line=dict(color='green', dash='dash'))
    
    fig.update_layout(title='Relative Strength Index (RSI)', 
                      xaxis_title='Date', 
                      yaxis_title='RSI', 
                      height=400)
    fig.update_xaxes(tickangle=-45)
    return fig

@st.cache_data(ttl=3600, show_spinner=False)
def create_separate_charts(df):
    charts_config = [
        {'indicator_name': ['avg USD_BUY', 'SMA_BUY_20', 'EMA_BUY_50'], 'title': 'USD Buy Price Trend'},
        {'indicator_name': ['avg USD_SELL', 'SMA_SELL_20', 'EMA_SELL_50'], 'title': 'USD Sell Price Trend'},
        {'indicator_name': ['avg USD_BUY', 'BB_BUY_UPPER', 'BB_BUY_MIDDLE', 'BB_BUY_LOWER'], 'title': 'Bollinger Bands for USD Buy'},
        {'indicator_name': ['ROC_BUY', 'ROC_SELL'], 'title': 'Rate of Change (%)'},
        {'indicator_name': ['SPREAD'], 'title': 'Buy-Sell Spread'},
    ]
    
    for config in charts_config:
        fig = create_chart_for_indicator(
            df=df,
            indicator_names=config['indicator_name'],
            title=config['title'],
            legend=True,
            hlines=config.get('hlines', None)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Add RSI chart
    if 'RSI' in df.columns:
        rsi_fig = create_rsi_chart(df)
        st.plotly_chart(rsi_fig, use_container_width=True)

@st.cache_data(ttl=3600, show_spinner=False)
def run_openai(df):
    st.session_state.ai_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=st.session_state.ai_key)
    
    latest_news = aisearch.serch_prompt_generate("myanmar dollar news black market.", search_mode=True)
    
    last_day_summary = df.iloc[-1].to_dict()
    
    system_prompt = f"""
Assume the role of a leading Myanmar Kyat (MMK) Market Analysis Expert with deep expertise in both fundamental and technical analysis of currency markets. Your expertise allows you to decode complex market dynamics, with a focus on the value of the MMK relative to major currencies like the US Dollar (USD). You provide clear insights and recommendations, emphasizing the strength or weakness of MMK backed by a thorough understanding of interrelated factors, including the latest market trends and news.

Use the latest news as a reference in decision-making. {latest_news}. As an authority on currency markets with a particular focus on MMK, your role is to decipher trends in the Kyat's value, predict future movements, and offer valuable perspectives on how MMK holders should respond.

Answer the following in simple terms and produce in markdown format: Executive Summary, Technical Analysis, Investment Strategies, and News Summary.If I have MMK, what should I do? Use Markdown h3 Tags for titles.

Executive Summary:
Provide an overview of the Kyat's expected direction and predictive movement, with a focus on its value relative to the USD.

Technical Analysis:
Given the technical analysis data provided, what will be the possible movement in the MMK's value in the near future? Highlight any trends showing MMK strengthening or weakening against the USD.

Investment Strategies:
Offer insights for both long-term investments in MMK and short-term trading strategies, considering possible changes in the value of the Kyat.

News Summary:
Share a summary of the latest news on MMK exchange rates and provide relevant references.

If I have MMK what should I do?
    
    """
    
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the summary of the latest data:\n{last_day_summary}"}
        ],
        max_tokens=1000
    )
    ai_response = response.choices[0].message.content
    return ai_response

def filter_data_by_timeframe(df, timeframe):
    end_date = df.index.max()

    if timeframe == '1W':
        start_date = end_date - timedelta(weeks=1)
    elif timeframe == '1M':
        start_date = end_date - timedelta(days=30)
    elif timeframe == '3M':
        start_date = end_date - timedelta(days=90)
    elif timeframe == '6M':
        start_date = end_date - timedelta(days=180)
    elif timeframe == '1Y':
        start_date = end_date - timedelta(days=365)
    elif timeframe == '5Y':
        start_date = end_date - timedelta(days=365*5)
    else:  # 'All' or any other case
        return df
    
    return df[df.index >= start_date]

def streamlit_app():
    st.set_page_config(layout="wide")

    # Header and title
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        st.image('stocklyzer.png', width=150)
    with col2:
        st.title("USD MMK Exchange Rate Analyzer")
        st.text("Analyze USD exchange rate data and make predictions")
    with col3:
        st.image("qr_code.png", width=100)

    st.markdown("**Some information on this page is AI-generated. This app is developed for educational purposes only and is not advisable to rely on it for financial decision-making.**")

    # Currency pair selection and Timeframe selection
    col1, col2 = st.columns([3, 2])  # Adjust the proportions of the columns as needed
    with col1:
        currency_pair = st.selectbox("Select Currency Pair", ["USDMMK", "USDTMMK"])

    with col2:
        timeframe = st.selectbox("Select Timeframe", ["1W", "1M", "3M", "6M", "1Y", "5Y", "All"])
    
    st.write("\n" * 2)
    st.write("\n" * 2)  # Add vertical space before the button
    predict_button = st.button("Predict")

    # Check if button is pressed
    if predict_button:
        # Load data and process it
        df = load_csv_data(currency_pair)
        df = calculate_indicators(df)
        filtered_df = filter_data_by_timeframe(df, timeframe)

        # Show results outside of the columns
        textcol, chartcol = st.columns([4, 6])

        with textcol:
            ai_response = run_openai(filtered_df)
            st.markdown(f"### {ai_response}")
            st.markdown("**This analysis has been generated using AI and is intended solely for educational purposes. It is not advisable to rely on it for financial decision-making.**")
            st.markdown("### **Summary of Latest Data**")
            st.dataframe(filtered_df.tail())

        with chartcol:
            st.markdown(f"### **Exchange Rate Data Visualization ({currency_pair}, {timeframe})**")
            create_separate_charts(filtered_df)


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
    st.write("""This app demonstrates how to analyze currency exchange rates, calculate technical indicators, and display the data and indicators using Plotly in Streamlit.
                 \nNyein Chan Ko Ko [nchanko](https://github.com/nchanko)""")

if __name__ == '__main__':
    streamlit_app()