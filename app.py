import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from model import run_prediction, STOCK_TICKERS, trading_backtest
import os

# Page config
st.set_page_config(
    page_title="Prediksi Saham Perbankan",
    page_icon="📈",
    layout="wide"
)

# Simple CSS
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
    }
    
    [data-testid="stSidebar"] {
        background-color: #f0f0f0;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1f1f1f !important;
    }
    
    p, span, div {
        color: #1f1f1f;
    }
    
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #45a049;
    }
    
    [data-testid="stMetricLabel"] {
        color: #666666 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #1f1f1f !important;
    }
    
    .stDataFrame {
        color: #1f1f1f;
    }
    
    .stMarkdown {
        color: #1f1f1f;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Sistem Prediksi Harga Saham Perbankan")
st.write("Prediksi harga saham menggunakan algoritma Random Forest")
st.write("---")

# CSV filename mapping
CSV_FILES = {
    'BBCA': 'Bank Central Asia Stock Price History.csv',
    'BBRI': 'Bank Rakyat Persero Stock Price History.csv',
    'BMRI': 'Bank Mandiri Stock Price History.csv',
    'BBNI': 'Bank Negar Stock Price History.csv',
    'BBTN': 'Bank Tabungan Negara Stock Price History.csv'
}


# Sidebar
with st.sidebar:
    st.header("Pengaturan Prediksi")
    
    # Stock selection
    stock_options = {
        "BBCA": "Bank Central Asia",
        "BBRI": "Bank Rakyat Indonesia", 
        "BMRI": "Bank Mandiri",
        "BBNI": "Bank Negara Indonesia",
        "BBTN": "Bank Tabungan Negara"
    }
    
    selected_stock = st.selectbox(
        "Pilih Saham:",
        options=list(stock_options.keys()),
        format_func=lambda x: f"{x} - {stock_options[x]}"
    )
    
    # Prediction days
    prediction_days = st.slider(
        "Prediksi Berapa Hari:",
        min_value=1,
        max_value=30,
        value=5
    )
    
    st.write("---")
    
    # Predict button
    analyze_btn = st.button("Jalankan Prediksi", type="primary")

# Main content
if analyze_btn:
    # Get CSV path
    csv_filename = CSV_FILES[selected_stock]
    csv_path = os.path.join('data', csv_filename)
    
    # Check if file exists
    if not os.path.exists(csv_path):
        st.error(f"File tidak ditemukan: {csv_path}")
    else:
        # Loading state
        with st.spinner(f"Memproses data dan melakukan prediksi {prediction_days} hari..."):
            # Run prediction
            ticker = STOCK_TICKERS[selected_stock]
            results = run_prediction(
                ticker, 
                prediction_days=prediction_days,
                csv_file=csv_path
            )
        
        # Check if successful
        if not results['success']:
            st.error(f"Terjadi kesalahan: {results.get('error', 'Unknown error')}")
        else:
            st.success("Prediksi berhasil dilakukan!")
            
            st.header("Hasil Prediksi Harga Saham")
            st.write("---")
            
            # Extract results
            metrics = results['model_metrics']
            current_price = results['current_price']
            future_predictions = results['future_predictions']
            future_dates = results['future_dates']
            
            # Top metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Saham", selected_stock)
            
            with col2:
                st.metric("Harga Saat Ini", f"Rp {current_price:,.0f}")
            
            with col3:
                avg_prediction = np.mean(future_predictions)
                change_pct = ((avg_prediction - current_price) / current_price) * 100
                st.metric(
                    f"Prediksi Rata-rata ({prediction_days} hari)",
                    f"Rp {avg_prediction:,.0f}",
                    f"{change_pct:+.2f}%"
                )
            
            with col4:
                st.metric("Akurasi Model", f"{metrics['accuracy']:.1f}%")
            
            st.write("")
            
            # Prediction table
            st.subheader("Detail Prediksi Per Hari")
            
            prediction_df = pd.DataFrame({
                'Tanggal': [d.strftime('%d %b %Y') for d in future_dates],
                'Harga Prediksi': [f"Rp {p:,.0f}" for p in future_predictions],
                'Perubahan dari Sekarang': [f"{((p-current_price)/current_price*100):+.2f}%" for p in future_predictions]
            })
            
            st.dataframe(prediction_df, use_container_width=True, hide_index=True)
            
            st.write("")
            
            # Chart
            st.subheader("Grafik Harga Historical vs Prediksi")
            
            fig = go.Figure()
            
            # Historical prices
            fig.add_trace(go.Scatter(
                x=results['recent_dates'],
                y=results['recent_prices'],
                mode='lines',
                name='Harga Historical',
                line=dict(color='blue', width=2)
            ))
            
            # Future predictions
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_predictions,
                mode='lines+markers',
                name='Prediksi',
                line=dict(color='green', width=2, dash='dash'),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                xaxis_title="Tanggal",
                yaxis_title="Harga (Rp)",
                hovermode='x unified',
                height=450
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("")
            
            # Model Performance
            st.subheader("Performa Model")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("MAE", f"Rp {metrics['mae']:,.0f}")
            
            with col2:
                st.metric("RMSE", f"Rp {metrics['rmse']:,.0f}")
            
            with col3:
                st.metric("MAPE", f"{metrics['mape']:.2f}%")
            
            with col4:
                st.metric("R² Score", f"{metrics['r2']:.4f}")
            
            st.write("")
            
            # Model info
            with st.expander("Informasi Model"):
                st.write(f"Jumlah Data Training: {results['training_samples']:,}")
                st.write(f"Algoritma: Random Forest (100 trees)")
                st.write(f"Jumlah Fitur: 22 indikator teknikal")
                
                st.write("")
                st.write("Top 5 Feature Importance:")
                top_features = results['feature_importance'].head(5)
                for idx, row in top_features.iterrows():
                    st.write(f"- {row['feature']}: {row['importance']:.4f}")
                # Test Set Performance
                st.markdown("---")
                st.write("Test Set Performance")
                st.write("Perbandingan prediksi model dengan harga actual pada test set")

                # Create chart
                test_fig = go.Figure()

                test_indices = list(range(len(results['test_actual'])))

                test_fig.add_trace(go.Scatter(
                    x=test_indices, 
                    y=results['test_actual'],
                    mode='lines', 
                    name='Actual Price',
                    line=dict(color='#2196F3', width=2)
                ))

                test_fig.add_trace(go.Scatter(
                    x=test_indices, 
                    y=results['test_predictions'],
                    mode='lines', 
                    name='Predicted Price',
                    line=dict(color='#4CAF50', width=2)
                ))

                test_fig.update_layout(
                    xaxis_title="Test Sample Index",
                    yaxis_title="Price (Rp)",
                    hovermode='x unified',
                    height=400,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#1f1f1f'),
                    xaxis=dict(gridcolor='#e0e0e0'),
                    yaxis=dict(gridcolor='#e0e0e0', tickformat=',.0f', tickprefix='Rp ')
                )

                st.plotly_chart(test_fig, use_container_width=True)

                st.write("")
            
            # Trading Backtest
            st.write("---")
            st.header("Simulasi Trading (Backtesting)")
            st.write("Simulasi strategi trading dengan Take Profit dan Stop Loss")
            
            with st.spinner("Menjalankan simulasi trading..."):
                trading_results = trading_backtest(
                    csv_path,
                    prediction_days=prediction_days,
                    initial_capital=10_000_000
                )
            
            if not trading_results['success']:
                st.error(f"Simulasi gagal: {trading_results.get('error')}")
            else:
                tr = trading_results
                summary = tr['summary']
                
                st.success("Simulasi trading selesai!")
                
                st.subheader("Ringkasan Hasil Trading")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Total Return",
                        f"{summary['total_return']:+.2f}%",
                        f"Rp {summary['total_profit']:,.0f}"
                    )
                
                with col2:
                    st.metric("Jumlah Transaksi", summary['total_trades'])
                
                with col3:
                    st.metric(
                        "Win Rate",
                        f"{summary['win_rate']:.1f}%",
                        f"{summary['winning_trades']} menang"
                    )
                
                col4, col5, col6 = st.columns(3)
                
                with col4:
                    st.metric(
                        "Rata-rata Profit per Trade",
                        f"{summary['avg_profit_per_trade_pct']:+.2f}%"
                    )
                
                with col5:
                    st.metric(
                        "Maximum Drawdown",
                        f"{summary['max_drawdown_pct']:.2f}%"
                    )
                
                with col6:
                    st.metric("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}")
                
                st.write("")
                
                # Exit breakdown
                st.subheader("Detail Exit Strategy")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Take Profit", summary['tp_exits'])
                
                with col2:
                    st.metric("Stop Loss", summary['sl_exits'])
                
                with col3:
                    st.metric("Time Exit", summary['time_exits'])
                
                with col4:
                    st.metric(
                        "vs Buy & Hold",
                        f"{summary['outperformance']:+.2f}%"
                    )
                
                st.write("")
                
                # Trade history
                st.subheader("Riwayat Transaksi")
                
                trades_df = pd.DataFrame(tr['trades'])
                
                if not trades_df.empty:
                    trades_display = trades_df[[
                        'trade_num', 'buy_date', 'buy_price', 'sell_date', 
                        'sell_price', 'exit_reason', 'profit', 'profit_pct'
                    ]].copy()
                    
                    trades_display['buy_date'] = pd.to_datetime(trades_display['buy_date']).dt.strftime('%Y-%m-%d')
                    trades_display['sell_date'] = pd.to_datetime(trades_display['sell_date']).dt.strftime('%Y-%m-%d')
                    
                    trades_display.columns = [
                        'No', 'Tanggal Beli', 'Harga Beli', 'Tanggal Jual', 
                        'Harga Jual', 'Alasan Exit', 'Profit (Rp)', 'Return (%)'
                    ]
                    
                    # Format numbers
                    for col in ['Harga Beli', 'Harga Jual', 'Profit (Rp)']:
                        trades_display[col] = trades_display[col].apply(lambda x: f"{x:,.0f}")
                    
                    trades_display['Return (%)'] = trades_display['Return (%)'].apply(lambda x: f"{x:+.2f}%")
                    
                    st.dataframe(trades_display, use_container_width=True, hide_index=True)
                    
                    # Portfolio value chart
                    st.subheader("Perkembangan Nilai Portfolio")
                    
                    portfolio_df = pd.DataFrame(tr['portfolio_value'])
                    
                    fig_portfolio = go.Figure()
                    
                    fig_portfolio.add_trace(go.Scatter(
                        x=portfolio_df['date'],
                        y=portfolio_df['value'],
                        mode='lines',
                        fill='tozeroy',
                        name='Nilai Portfolio'
                    ))
                    
                    fig_portfolio.add_hline(
                        y=summary['initial_capital'],
                        line_dash="dash",
                        annotation_text="Modal Awal"
                    )
                    
                    fig_portfolio.update_layout(
                        xaxis_title="Tanggal",
                        yaxis_title="Nilai Portfolio (Rp)",
                        height=400
                    )
                    
                    st.plotly_chart(fig_portfolio, use_container_width=True)

# Landing page
else:
    # Auto-load all stocks prediction on page load
    st.header("Prediksi Harga Saham Hari Ini")
    st.write("Prediksi harga saham untuk besok (1 hari ke depan)")

    # Create columns for all stocks
    cols = st.columns(5)
    all_predictions = []
    stock_options = {
        "BBCA": "Bank Central Asia",
        "BBRI": "Bank Rakyat Indonesia", 
        "BMRI": "Bank Mandiri",
        "BBNI": "Bank Negara Indonesia",
        "BBTN": "Bank Tabungan Negara"
    }

    for idx, (stock_code, stock_name) in enumerate(stock_options.items()):
        csv_filename = CSV_FILES[stock_code]
        csv_path = os.path.join('data', csv_filename)
        
        if os.path.exists(csv_path):
            try:
                # Run prediction for 1 day
                ticker = STOCK_TICKERS[stock_code]
                results = run_prediction(
                    ticker, 
                    prediction_days=1,
                    csv_file=csv_path
                )
                
                if results['success']:
                    current_price = results['current_price']
                    tomorrow_price = results['future_predictions'][0]
                    change_pct = ((tomorrow_price - current_price) / current_price) * 100
                    
                    with cols[idx]:
                        st.metric(
                            label=f"{stock_code}",
                            value=f"Rp {current_price:,.0f}",
                            delta=f"{change_pct:+.2f}%",
                            help=stock_name
                        )
                        st.caption(f"Prediksi: Rp {tomorrow_price:,.0f}")
            except:
                with cols[idx]:
                    st.metric(
                        label=f"{stock_code}",
                        value="Error",
                        help=stock_name
                    )

    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Pilih saham dan pengaturan prediksi di sidebar, lalu klik 'Jalankan Prediksi'")
    
    st.subheader("Fitur Sistem")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Prediksi Harga**")
        st.write("Prediksi 1-30 hari ke depan")
    
    with col2:
        st.write("**Data Real**")
        st.write("5 tahun data historis")
    
    with col3:
        st.write("**Simulasi Trading**")
        st.write("Backtest dengan TP/SL")
    
    st.write("")
    
    st.subheader("Spesifikasi Teknis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model:**")
        st.write("- Algoritma: Random Forest")
        st.write("- Jumlah Trees: 100")
        st.write("- Fitur: 22 indikator teknikal")
    
    with col2:
        st.write("**Indikator:**")
        st.write("- Moving Average (SMA, EMA)")
        st.write("- MACD")
        st.write("- RSI")
        st.write("- Bollinger Bands")
    
    

# Footer
st.write("---")
st.caption("Sistem Prediksi Saham Perbankan Indonesia | Data: Investing.com | Model: Random Forest")