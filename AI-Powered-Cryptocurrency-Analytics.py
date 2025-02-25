import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Linear Algebra Project")

# Function to load data based on coin selection
def load_data(coin):
    file_paths = {
        "Bitcoin": "E:/Stats_project/coin_Bitcoin.csv",
        "Ethereum": "E:/Stats_project/coin_Ethereum.csv",
        "Binance": "E:/Stats_project/coin_BinanceCoin.csv",
        "Dogecoin": "E:/Stats_project/coin_Dogecoin.csv",
        "USDC": "E:/Stats_project/coin_USDCoin.csv",
        "Cardano": "E:/Stats_project/coin_Cardano.csv",
        "XRP": "E:/Stats_project/coin_XRP.csv",
        "Solana": "E:/Stats_project/coin_Solana.csv"
    }
    file_path = file_paths.get(coin)
    if file_path:
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        return data
    else:
        st.error("Invalid coin selection")
        return None

# Visualization functions
def plot_price_movement(data):
    st.subheader("Price Movement")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Date'], data['High'], label='High', alpha=0.7)
    ax.plot(data['Date'], data['Low'], label='Low', alpha=0.7)
    ax.plot(data['Date'], data['Close'], label='Close', linewidth=2)
    ax.set_title('Price Movement')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def plot_time_series_analysis(data):
    st.subheader("Time Series Analysis: Open and Close Prices")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Date'], data['Open'], label='Open', alpha=0.7)
    ax.plot(data['Date'], data['Close'], label='Close', alpha=0.7)
    ax.set_title('Open and Close Prices Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def plot_candlestick(data):
    st.subheader("Candlestick Pattern Analysis")
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.6
    width2 = 0.1

    for i in range(len(data)):
        color = 'g' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'r'
        ax.bar(i, data['Close'].iloc[i] - data['Open'].iloc[i], bottom=data['Open'].iloc[i], color=color, width=width)
        ax.bar(i, data['High'].iloc[i] - data['Low'].iloc[i], bottom=data['Low'].iloc[i], color=color, width=width2)

    ax.set_title('Candlestick Pattern')
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Price (USD)')
    st.pyplot(fig)

def plot_correlation_heatmap(data):
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = data[['High', 'Low', 'Open', 'Close', 'Marketcap']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)

def plot_svd_analysis(X_svd, data):
    st.subheader("SVD Component Analysis")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_svd[:, 0], X_svd[:, 1], c=data['Close'], cmap='viridis', edgecolor='k', s=50)
    ax.set_title('SVD Component Analysis')
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    fig.colorbar(scatter, ax=ax, label='Closing Price')
    st.pyplot(fig)

def plot_pca_analysis(X_pca, data):
    st.subheader("PCA Component Analysis")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=data['Close'], cmap='coolwarm', edgecolor='k', s=50)
    ax.set_title('PCA Component Analysis')
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    fig.colorbar(scatter, ax=ax, label='Closing Price')
    st.pyplot(fig)

def plot_anomaly_detection(data):
    st.subheader("Anomaly Detection")
    z_scores = np.abs((data['Close'] - data['Close'].mean()) / data['Close'].std())
    anomalies = data[z_scores > 3]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Date'], data['Close'], label='Close', alpha=0.7)
    ax.scatter(anomalies['Date'], anomalies['Close'], color='red', label='Anomalies', s=50)
    ax.set_title('Anomaly Detection in Closing Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def plot_control_charts(data):
    st.subheader("Control Charts")

    # Price Control Chart
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 16))

    # 1. Shewhart Control Chart
    close_mean = data['Close'].mean()
    close_std = data['Close'].std()

    ax1.plot(data['Date'], data['Close'], marker='o', markersize=3, label='Close Price')
    ax1.axhline(y=close_mean, color='g', linestyle='--', label='Mean')
    ax1.axhline(y=close_mean + 3 * close_std, color='r', linestyle='--', label='UCL')
    ax1.axhline(y=close_mean - 3 * close_std, color='r', linestyle='--', label='LCL')
    ax1.set_title('Shewhart Control Chart')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True)
    ax1.tick_params(axis='x', rotation=45)

    # 2. EWMA Control Chart
    lambda_ewma = 0.2
    ewma = [data['Close'].iloc[0]]
    for i in range(1, len(data)):
        ewma.append(lambda_ewma * data['Close'].iloc[i] + (1 - lambda_ewma) * ewma[-1])
    ewma_std = close_std * (lambda_ewma / (2 - lambda_ewma)) ** 0.5
    ewma_upper = [close_mean + 3 * ewma_std for _ in ewma]
    ewma_lower = [close_mean - 3 * ewma_std for _ in ewma]

    ax2.plot(data['Date'], ewma, marker='o', markersize=3, label='EWMA')
    ax2.plot(data['Date'], ewma_upper, color='r', linestyle='--', label='UCL (EWMA)')
    ax2.plot(data['Date'], ewma_lower, color='r', linestyle='--', label='LCL (EWMA)')
    ax2.axhline(y=close_mean, color='g', linestyle='--', label='Mean')
    ax2.set_title('EWMA Control Chart')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('EWMA (USD)')
    ax2.legend()
    ax2.grid(True)
    ax2.tick_params(axis='x', rotation=45)

    # 3. CUSUM Control Chart
    target = close_mean
    k = 0.5 * close_std
    h = 5 * close_std
    S_pos = [0]
    S_neg = [0]

    for i in range(1, len(data)):
        deviation = data['Close'].iloc[i] - target
        S_pos.append(max(0, S_pos[-1] + deviation - k))
        S_neg.append(min(0, S_neg[-1] + deviation + k))

    ax3.plot(data['Date'], S_pos, color='b', label='CUSUM Positive')
    ax3.plot(data['Date'], S_neg, color='r', label='CUSUM Negative')
    ax3.axhline(y=h, color='g', linestyle='--', label='Upper Control Limit')
    ax3.axhline(y=-h, color='g', linestyle='--', label='Lower Control Limit')
    ax3.set_title('CUSUM Control Chart')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Cumulative Sum (USD)')
    ax3.legend()
    ax3.grid(True)
    ax3.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)

def plot_price_distribution(data):
    st.subheader("Price Distribution")
    plt.figure(figsize=(12, 6))
    sns.histplot(data=data, x='Close', kde=True)
    plt.title('Closing Price Distribution')
    plt.xlabel('Price (USD)')
    plt.ylabel('Frequency')
    plt.grid(True)
    st.pyplot(plt.gcf())

# Streamlit application
def main():
    st.title("Cryptocurrency Price Analysis")

    # Sidebar for coin selection
    st.sidebar.title("Select Cryptocurrency")
    coin = st.sidebar.selectbox(
        "Choose a cryptocurrency:",
        ["Bitcoin", "Ethereum", "Binance", "Dogecoin", "USDC","XRP","Solana",""]
    )

    # Load data based on coin selection
    data = load_data(coin)
    if data is not None:
        # Tabs for visualization selection
        tabs = st.tabs([
            "Price Movement",
            "Time Series Analysis",
            "Candlestick Pattern",
            "Correlation Heatmap",
            "SVD Component Analysis",
            "PCA Component Analysis",
            "Anomaly Detection",
            "Control Charts",
            "Price Distribution"
        ])

        # Prepare data for dimensionality reduction
        price_columns = ['High', 'Low', 'Open', 'Close']
        X = data[price_columns]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform SVD and PCA
        svd = TruncatedSVD(n_components=2)
        X_svd = svd.fit_transform(X_scaled)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # Render visualizations in respective tabs
        with tabs[0]:
            plot_price_movement(data)
        with tabs[1]:
            plot_time_series_analysis(data)
        with tabs[2]:
            plot_candlestick(data)
        with tabs[3]:
            plot_correlation_heatmap(data)
        with tabs[4]:
            plot_svd_analysis(X_svd, data)
        with tabs[5]:
            plot_pca_analysis(X_pca, data)
        with tabs[6]:
            plot_anomaly_detection(data)
        with tabs[7]:
            plot_control_charts(data)
        with tabs[8]:
            plot_price_distribution(data)
    else:
        st.error("Unable to load data.")

if __name__ == "__main__":
    main()
