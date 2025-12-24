import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    
    def __init__(self, ticker, prediction_days=5, n_estimators=100, random_state=42):
        self.ticker = ticker
        self.prediction_days = prediction_days
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_names = []
        
    def load_from_csv(self, csv_file):
        """Load data dari CSV (Investing.com format)"""
        try:
            print(f"📂 Loading data from CSV...")
            
            # Read CSV
            df = pd.read_csv(csv_file)
            
            # Clean columns
            df.columns = [col.lower().strip() for col in df.columns]
            print(f"   Columns: {list(df.columns)}")
            
            # Rename Investing.com columns
            if 'price' in df.columns:
                df.rename(columns={'price': 'close'}, inplace=True)
            if 'vol.' in df.columns:
                df.rename(columns={'vol.': 'volume'}, inplace=True)
            
            # Parse date
            df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False)
            
            # Clean prices (remove commas)
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns and df[col].dtype == 'object':
                    df[col] = df[col].str.replace(',', '').astype(float)
            
            # Clean volume (handle M, K, B suffix)
            if 'volume' in df.columns and df['volume'].dtype == 'object':
                def parse_volume(vol):
                    if pd.isna(vol) or vol == '-':
                        return 0
                    vol = str(vol).strip().replace(',', '')
                    if 'M' in vol:
                        return int(float(vol.replace('M', '')) * 1_000_000)
                    elif 'K' in vol:
                        return int(float(vol.replace('K', '')) * 1_000)
                    elif 'B' in vol:
                        return int(float(vol.replace('B', '')) * 1_000_000_000)
                    else:
                        return int(float(vol))
                
                df['volume'] = df['volume'].apply(parse_volume)
            
            # Sort by date (oldest first)
            df = df.sort_values('date').reset_index(drop=True)
            
            # Remove NaN
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            
            print(f"   ✅ Loaded {len(df)} days")
            print(f"   📅 {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
            print(f"   💰 Price range: Rp {df['close'].min():,.0f} - Rp {df['close'].max():,.0f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators"""
        df = df.copy()
        
        # Moving Averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        
        # EMA
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(10)
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(window=10).mean()
        
        # Lag features
        for i in range(1, 6):
            df[f'close_lag_{i}'] = df['close'].shift(i)
        
        return df
    
    def create_sequences(self, df, target_days_ahead=5):
        """Create training sequences"""
        feature_cols = [
            'open', 'high', 'low', 'volume',
            'sma_5', 'sma_10', 'sma_20',
            'ema_12', 'ema_26', 'macd', 'macd_signal',
            'rsi', 'bb_middle', 'bb_upper', 'bb_lower',
            'momentum', 'volume_sma',
            'close_lag_1', 'close_lag_2', 'close_lag_3', 
            'close_lag_4', 'close_lag_5'
        ]
        
        self.feature_names = feature_cols
        
        # Clean NaN
        df_clean = df.dropna()
        
        # Create target: harga N hari ke depan
        df_clean = df_clean.copy()
        df_clean['target'] = df_clean['close'].shift(-target_days_ahead)
        
        # Remove last N rows (no target)
        df_clean = df_clean[:-target_days_ahead].dropna()
        
        X = df_clean[feature_cols].values
        y = df_clean['target'].values
        
        return X, y, df_clean
    
    def train_model(self, X_train, y_train):
        """Train Random Forest"""
        print(f"🎓 Training with {len(X_train)} samples...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        print(f"✅ Training complete!")
    
    def predict_future(self, df_clean, days_ahead=5, max_daily_change=0.03):
        """Predict future prices with realistic constraints"""
        predictions = []
        current_price = df_clean['close'].iloc[-1]
        
        # Get latest features
        latest_features = df_clean.iloc[-1:][self.feature_names].values
        latest_scaled = self.scaler.transform(latest_features)
        
        # Predict first day
        pred_price = self.model.predict(latest_scaled)[0]
        
        # Apply constraint
        max_up = current_price * (1 + max_daily_change)
        max_down = current_price * (1 - max_daily_change)
        pred_price = np.clip(pred_price, max_down, max_up)
        
        predictions.append(pred_price)
        
        # Predict remaining days
        for day in range(1, days_ahead):
            prev_price = predictions[-1]
            
            # Small variation
            noise_factor = 1 + np.random.randn() * 0.001
            varied_features = latest_features * noise_factor
            varied_scaled = self.scaler.transform(varied_features)
            
            pred_price = self.model.predict(varied_scaled)[0]
            
            # Apply constraint
            max_up = prev_price * (1 + max_daily_change)
            max_down = prev_price * (1 - max_daily_change)
            pred_price = np.clip(pred_price, max_down, max_up)
            
            predictions.append(pred_price)
        
        return predictions
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model"""
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        accuracy = np.mean(np.abs((y_test - predictions) / y_test) < 0.05) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'accuracy': accuracy,
            'predictions': predictions,
            'actual': y_test
        }
    
    def get_feature_importance(self):
        """Get feature importance"""
        if self.model is None:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


def run_prediction(ticker, prediction_days=5, csv_file=None):
    """Main function: Train model & predict future"""
    try:
        predictor = StockPredictor(ticker, prediction_days=prediction_days)
        
        # 1. Load data from CSV
        if csv_file is None:
            return {'success': False, 'error': 'CSV file path required'}
        
        df = predictor.load_from_csv(csv_file)
        
        if df is None or len(df) < 100:
            return {'success': False, 'error': 'Insufficient data (need at least 100 days)'}
        
        # 2. Calculate indicators
        print("📊 Calculating technical indicators...")
        df = predictor.calculate_technical_indicators(df)
        
        # 3. Create sequences
        print("🔧 Creating training sequences...")
        X, y, df_clean = predictor.create_sequences(df, target_days_ahead=prediction_days)
        
        if len(X) < 100:
            return {'success': False, 'error': 'Not enough data after feature engineering'}
        
        # 4. Split train/test (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 5. Train model
        predictor.train_model(X_train, y_train)
        
        # 6. Evaluate
        print("📈 Evaluating model...")
        test_metrics = predictor.evaluate_model(X_test, y_test)
        
        # 7. Predict future (with 3% daily limit)
        print(f"🔮 Predicting {prediction_days} days ahead...")
        future_predictions = predictor.predict_future(
            df_clean, 
            days_ahead=prediction_days,
            max_daily_change=0.03
        )
        
        # 8. Generate future dates (skip weekends)
        last_date = df['date'].iloc[-1]
        future_dates = []
        current_date = last_date
        while len(future_dates) < prediction_days:
            current_date += timedelta(days=1)
            if current_date.weekday() < 5:
                future_dates.append(current_date)
        
        # 9. Get recent historical (FIXED: use original df, not df_clean)
        recent_days = min(30, len(df))
        recent_data = df.tail(recent_days)[['date', 'close']]
        recent_dates = recent_data['date'].values
        recent_prices = recent_data['close'].values
        
        print(f"📊 Historical data: {len(recent_dates)} days")
        print(f"   Last date: {recent_dates[-1]}")
        print(f"🔮 Future predictions: {len(future_predictions)} days")
        
        # 10. Calculate stats
        current_price = df['close'].iloc[-1]
        avg_predicted = np.mean(future_predictions)
        predicted_change = ((avg_predicted - current_price) / current_price) * 100
        
        # 11. Return results
        results = {
            'success': True,
            'ticker': ticker,
            'training_samples': len(X_train),
            'model_metrics': {
                'mae': test_metrics['mae'],
                'rmse': test_metrics['rmse'],
                'mape': test_metrics['mape'],
                'r2': test_metrics['r2'],
                'accuracy': test_metrics['accuracy']
            },
            'current_price': current_price,
            'future_predictions': future_predictions,
            'future_dates': future_dates,
            'predicted_change_pct': predicted_change,
            'recent_dates': recent_dates,
            'recent_prices': recent_prices,
            'test_predictions': test_metrics['predictions'][-20:],
            'test_actual': test_metrics['actual'][-20:],
            'feature_importance': predictor.get_feature_importance()
        }
        
        print("✅ Prediction complete!")
        return results
        
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def trading_backtest(csv_file, prediction_days=5, initial_capital=10_000_000):
    """
    Automatic trading backtest dengan TP/SL 1:1
    
    Strategy:
    - BUY: Jika predicted gain > 0.5%
    - TP (Take Profit): Sesuai prediction gain
    - SL (Stop Loss): 1:1 ratio dengan TP
    - Exit: Hit TP/SL atau holding period habis
    """
    try:
        print(f"\n💰 Running Automatic Trading Backtest...")
        print(f"Strategy: TP/SL 1:1 based on prediction")
        
        # Load & train model
        predictor = StockPredictor('TRADING', prediction_days=prediction_days)
        df = predictor.load_from_csv(csv_file)
        
        if df is None or len(df) < 500:
            return {'success': False, 'error': 'Need at least 500 days'}
        
        df = predictor.calculate_technical_indicators(df)
        X, y, df_clean = predictor.create_sequences(df, target_days_ahead=prediction_days)
        
        # Train
        train_size = int(len(X) * 0.6)
        X_train, y_train = X[:train_size], y[:train_size]
        
        predictor.feature_names = [
            'open', 'high', 'low', 'volume',
            'sma_5', 'sma_10', 'sma_20',
            'ema_12', 'ema_26', 'macd', 'macd_signal',
            'rsi', 'bb_middle', 'bb_upper', 'bb_lower',
            'momentum', 'volume_sma',
            'close_lag_1', 'close_lag_2', 'close_lag_3', 
            'close_lag_4', 'close_lag_5'
        ]
        predictor.train_model(X_train, y_train)
        
        # Trading simulation
        capital = initial_capital
        holdings = 0
        entry_price = 0
        tp_price = 0
        sl_price = 0
        trades = []
        portfolio_value = []
        
        position_open = False
        entry_idx = 0
        entry_date = None
        predicted_target = 0
        
        test_start = train_size
        
        for i in range(test_start, len(df_clean) - prediction_days, 3):
            
            current_date = df_clean.iloc[i]['date']
            current_price = df_clean.iloc[i]['close']
            high_price = df_clean.iloc[i]['high']
            low_price = df_clean.iloc[i]['low']
            
            # Check open position
            if position_open:
                # Check TP (using high price)
                if high_price >= tp_price:
                    exit_value = holdings * tp_price
                    entry_value = holdings * entry_price
                    profit = exit_value - entry_value  # ✅ Compare with entry!
                    profit_pct = (profit / entry_value) * 100
                    
                    capital = exit_value

                    
                    print(f"  ✅ TP Hit! Exit @ Rp {tp_price:,.0f} | Profit: {profit_pct:+.2f}%")
                    
                    trades.append({
                        'trade_num': len(trades) + 1,
                        'buy_date': entry_date,
                        'buy_price': entry_price,
                        'sell_date': current_date,
                        'sell_price': tp_price,
                        'predicted_price': predicted_target,
                        'tp_price': tp_price,
                        'sl_price': sl_price,
                        'exit_reason': 'TP',
                        'shares': holdings,
                        'profit': profit,
                        'profit_pct': profit_pct
                    })
                    
                    holdings = 0
                    position_open = False
                    
                # Check SL (using low price)
                elif low_price <= sl_price:
                    exit_value = holdings * sl_price
                    entry_value = holdings * entry_price
                    profit = exit_value - entry_value  # ✅ Should be NEGATIVE!
                    profit_pct = (profit / entry_value) * 100
                    
                    capital = exit_value

                    
                    print(f"  ❌ SL Hit! Exit @ Rp {sl_price:,.0f} | Loss: {profit_pct:+.2f}%")
                    
                    trades.append({
                        'trade_num': len(trades) + 1,
                        'buy_date': entry_date,
                        'buy_price': entry_price,
                        'sell_date': current_date,
                        'sell_price': sl_price,
                        'predicted_price': predicted_target,
                        'tp_price': tp_price,
                        'sl_price': sl_price,
                        'exit_reason': 'SL',
                        'shares': holdings,
                        'profit': profit,
                        'profit_pct': profit_pct
                    })
                    
                    holdings = 0
                    position_open = False
                
                # Time exit
                elif i - entry_idx >= prediction_days:
                    exit_value = holdings * current_price
                    entry_value = holdings * entry_price
                    profit = exit_value - entry_value  # ✅ Correct!
                    profit_pct = (profit / entry_value) * 100
                    
                    capital = exit_value

                    
                    print(f"  ⏰ Time Exit @ Rp {current_price:,.0f} | Result: {profit_pct:+.2f}%")
                    
                    trades.append({
                        'trade_num': len(trades) + 1,
                        'buy_date': entry_date,
                        'buy_price': entry_price,
                        'sell_date': current_date,
                        'sell_price': current_price,
                        'predicted_price': predicted_target,
                        'tp_price': tp_price,
                        'sl_price': sl_price,
                        'exit_reason': 'Time',
                        'shares': holdings,
                        'profit': profit,
                        'profit_pct': profit_pct
                    })
                    
                    holdings = 0
                    position_open = False
            
            # Look for entry
            if not position_open and len(trades) < 30:
                
                features = df_clean.iloc[i:i+1][predictor.feature_names].values
                features_scaled = predictor.scaler.transform(features)
                predicted_price = predictor.model.predict(features_scaled)[0]
                
                predicted_gain_pct = ((predicted_price - current_price) / current_price) * 100
                
                # BUY signal
                if predicted_gain_pct > 0.5:
                    shares = capital / current_price
                    holdings = shares
                    entry_price = current_price
                    entry_date = current_date
                    entry_idx = i
                    predicted_target = predicted_price
                    
                    # TP/SL 1:1
                    tp_price = entry_price * (1 + predicted_gain_pct / 100)
                    sl_price = entry_price * (1 - predicted_gain_pct / 100)
                    
                    capital = 0
                    position_open = True
                    
                    print(f"  📈 BUY {shares:.0f} shares @ Rp {entry_price:,.0f}")
                    print(f"     Predicted: {predicted_gain_pct:+.2f}% | TP: {tp_price:,.0f} | SL: {sl_price:,.0f}")
            
            # Track portfolio
            if holdings > 0:
                portfolio_val = holdings * current_price
            else:
                portfolio_val = capital
            
            portfolio_value.append({
                'date': current_date,
                'value': portfolio_val
            })
        
        # Final exit
        if holdings > 0:
            final_price = df_clean.iloc[-1]['close']
            exit_value = holdings * final_price
            entry_value = holdings * entry_price
            profit = exit_value - entry_value  # ✅ Correct!
            profit_pct = (profit / entry_value) * 100
            
            capital = exit_value
            
            trades.append({
                'trade_num': len(trades) + 1,
                'buy_date': entry_date,
                'buy_price': entry_price,
                'sell_date': final_date,
                'sell_price': final_price,
                'predicted_price': predicted_target,
                'tp_price': tp_price,
                'sl_price': sl_price,
                'exit_reason': 'Final',
                'shares': holdings,
                'profit': profit,
                'profit_pct': (profit / initial_capital) * 100
            })
        
        # Metrics
        total_return = ((capital - initial_capital) / initial_capital) * 100
        
        if trades:
            winning_trades = [t for t in trades if t['profit'] > 0]
            win_rate = (len(winning_trades) / len(trades)) * 100
            avg_profit = np.mean([t['profit'] for t in trades])
            avg_profit_pct = np.mean([t['profit_pct'] for t in trades])
            
            tp_exits = len([t for t in trades if t['exit_reason'] == 'TP'])
            sl_exits = len([t for t in trades if t['exit_reason'] == 'SL'])
            time_exits = len([t for t in trades if t['exit_reason'] == 'Time'])
        else:
            winning_trades = []
            win_rate = 0
            avg_profit = 0
            avg_profit_pct = 0
            tp_exits = sl_exits = time_exits = 0
        
        buy_hold_return = ((df_clean.iloc[-1]['close'] - df_clean.iloc[test_start]['close']) / 
                          df_clean.iloc[test_start]['close']) * 100
        
         # 1. Average Profit per Trade
        avg_profit_per_trade = avg_profit if trades else 0
        avg_profit_per_trade_pct = avg_profit_pct if trades else 0
        
        # 2. Maximum Drawdown
        portfolio_values = [p['value'] for p in portfolio_value]
        peak = initial_capital
        max_drawdown = 0
        max_drawdown_pct = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = peak - value
            drawdown_pct = (drawdown / peak) * 100
            if drawdown_pct > max_drawdown_pct:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct
        
        # 3. Sharpe Ratio (annualized)
        if trades and len(trades) > 1:
            returns = [t['profit_pct'] for t in trades]
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Assume 252 trading days per year
            # trading_days_per_year / avg_holding_period = trades_per_year
            avg_holding_days = prediction_days
            trades_per_year = 252 / avg_holding_days
            
            # Annualized return and volatility
            annualized_return = mean_return * trades_per_year
            annualized_volatility = std_return * np.sqrt(trades_per_year)
            
            # Risk-free rate (assume 6% annual for Indonesia)
            risk_free_rate = 6.0
            
            if annualized_volatility > 0:
                sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # 4. Total number of trades
        total_trades = len(trades)
        
        print(f"\n✅ Backtest Complete!")
        print(f"Return: {total_return:+.2f}% | Win Rate: {win_rate:.1f}%")
        print(f"TP: {tp_exits} | SL: {sl_exits} | Time: {time_exits}")
        print(f"Max Drawdown: {max_drawdown_pct:.2f}% | Sharpe: {sharpe_ratio:.2f}")
        
        # Return Value
        return {
            'success': True,
            'summary': {
                'initial_capital': initial_capital,
                'final_capital': capital,
                'total_return': total_return,
                'total_profit': capital - initial_capital,
                'num_trades': len(trades),
                'winning_trades': len(winning_trades),
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_profit_pct': avg_profit_pct,
                'tp_exits': tp_exits,
                'sl_exits': sl_exits,
                'time_exits': time_exits,
                'buy_hold_return': buy_hold_return,
                'outperformance': total_return - buy_hold_return,
                'avg_profit_per_trade': avg_profit_per_trade,
                'avg_profit_per_trade_pct': avg_profit_per_trade_pct,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown_pct,
                'sharpe_ratio': sharpe_ratio,
                'total_trades': total_trades
            },
            'trades': trades,
            'portfolio_value': portfolio_value
        }
        
    except Exception as e:
        print(f"❌ Backtest error: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


# Stock ticker mapping
STOCK_TICKERS = {
    'BBCA': 'BBCA.JK',
    'BBRI': 'BBRI.JK',
    'BMRI': 'BMRI.JK',
    'BBNI': 'BBNI.JK',
    'BBTN': 'BBTN.JK'
}