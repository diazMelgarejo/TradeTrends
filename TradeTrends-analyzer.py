#!/usr/bin/env python3
"""
Trade Trends Analyzer - Open Source TrendSpider Alternative
A free alternative focused on Bitcoin and Ethereum analysis
"""

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import talib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TradeTrendsAnalyzer:
    def __init__(self):
        self.data = {}
        self.signals = {}

    def fetch_crypto_data(self, symbol, period="1y", interval="1d"):
        """Fetch cryptocurrency data using Yahoo Finance"""
        try:
            ticker = yf.Ticker(f"{symbol}-USD")
            data = ticker.history(period=period, interval=interval)
            self.data[symbol] = data
            print(f"✅ Downloaded {symbol} data: {len(data)} candles")
            return data
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
            return None

    def calculate_indicators(self, symbol):
        """Calculate technical indicators matching your preferences"""
        if symbol not in self.data:
            return None

        data = self.data[symbol].copy()
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        volume = data['Volume'].values

        # Bollinger Bands (your preference)
        data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(
            close, timeperiod=20, nbdevup=2, nbdevdn=2
        )

        # Multiple EMAs (your preference: 20, 50, 100, 200)
        data['EMA_20'] = talib.EMA(close, timeperiod=20)
        data['EMA_50'] = talib.EMA(close, timeperiod=50) 
        data['EMA_100'] = talib.EMA(close, timeperiod=100)
        data['EMA_200'] = talib.EMA(close, timeperiod=200)

        # RSI for overbought/oversold
        data['RSI'] = talib.RSI(close, timeperiod=14)

        # MACD for momentum
        data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(close)

        # Volume analysis
        data['Volume_SMA'] = talib.SMA(volume.astype(float), timeperiod=20)
        data['Volume_Ratio'] = volume / data['Volume_SMA']

        # Supertrend (approximation)
        hl2 = (high + low) / 2
        atr = talib.ATR(high, low, close, timeperiod=10)
        data['ATR'] = atr

        self.data[symbol] = data
        return data

    def detect_sideways_market(self, symbol, lookback=20, threshold=0.05):
        """Detect sideways markets for Hidden Game strategy"""
        if symbol not in self.data:
            return False

        data = self.data[symbol].tail(lookback)
        price_range = (data['High'].max() - data['Low'].min()) / data['Close'].mean()

        return price_range < threshold

    def analyze_whale_signals(self, symbol, volume_threshold=1.5):
        """Detect potential whale activity through volume analysis"""
        if symbol not in self.data:
            return []

        data = self.data[symbol]
        signals = []

        # High volume + price action signals
        high_volume_days = data[data['Volume_Ratio'] > volume_threshold]

        for idx, row in high_volume_days.iterrows():
            price_change = (row['Close'] - row['Open']) / row['Open'] * 100

            signal_type = "accumulation" if abs(price_change) < 2 and row['Volume_Ratio'] > 2 else "breakout"

            signals.append({
                'date': idx,
                'type': signal_type,
                'volume_ratio': row['Volume_Ratio'],
                'price_change': price_change
            })

        return signals[-10:]  # Return last 10 signals

    def generate_entry_signals(self, symbol):
        """Generate entry signals based on your stored preferences"""
        if symbol not in self.data:
            return []

        data = self.data[symbol]
        signals = []

        latest = data.iloc[-1]
        prev = data.iloc[-2]

        # EMA alignment check (Golden Cross pattern)
        ema_bullish = (latest['EMA_20'] > latest['EMA_50'] > 
                      latest['EMA_100'] > latest['EMA_200'])

        # Bollinger Band signals
        bb_squeeze = (latest['BB_upper'] - latest['BB_lower']) / latest['BB_middle'] < 0.1
        bb_expansion = (latest['BB_upper'] - latest['BB_lower']) > (prev['BB_upper'] - prev['BB_lower'])

        # RSI oversold/overbought
        rsi_oversold = latest['RSI'] < 30
        rsi_overbought = latest['RSI'] > 70

        # Volume confirmation
        volume_spike = latest['Volume_Ratio'] > 1.2

        # Generate signals
        if ema_bullish and rsi_oversold and volume_spike:
            signals.append("STRONG_BUY - EMA alignment + oversold + volume")
        elif bb_expansion and volume_spike and not rsi_overbought:
            signals.append("BUY - BB expansion + volume confirmation")
        elif rsi_overbought and not ema_bullish:
            signals.append("SELL - Overbought in weak trend")
        elif bb_squeeze:
            signals.append("WATCH - Bollinger squeeze, breakout imminent")

        return signals

    def create_advanced_chart(self, symbol):
        """Create TrendSpider-like advanced chart"""
        if symbol not in self.data:
            return None

        data = self.data[symbol].tail(100)  # Last 100 candles

        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} Price & Indicators', 'RSI', 'MACD', 'Volume'),
            row_width=[0.2, 0.1, 0.1, 0.1]
        )

        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=f'{symbol}',
            increasing_line_color='#00ff00',
            decreasing_line_color='#ff0000'
        ), row=1, col=1)

        # Bollinger Bands
        fig.add_trace(go.Scatter(
            x=data.index, y=data['BB_upper'],
            name='BB Upper', line=dict(color='rgba(255,0,0,0.3)')
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=data.index, y=data['BB_lower'],
            name='BB Lower', line=dict(color='rgba(255,0,0,0.3)'),
            fill='tonexty'
        ), row=1, col=1)

        # EMAs
        colors = ['orange', 'blue', 'green', 'purple']
        emas = ['EMA_20', 'EMA_50', 'EMA_100', 'EMA_200']
        for ema, color in zip(emas, colors):
            fig.add_trace(go.Scatter(
                x=data.index, y=data[ema],
                name=ema, line=dict(color=color, width=1)
            ), row=1, col=1)

        # RSI
        fig.add_trace(go.Scatter(
            x=data.index, y=data['RSI'],
            name='RSI', line=dict(color='purple')
        ), row=2, col=1)

        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        # MACD
        fig.add_trace(go.Scatter(
            x=data.index, y=data['MACD'],
            name='MACD', line=dict(color='blue')
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=data.index, y=data['MACD_signal'],
            name='Signal', line=dict(color='red')
        ), row=3, col=1)

        # Volume
        fig.add_trace(go.Bar(
            x=data.index, y=data['Volume'],
            name='Volume', marker_color='rgba(0,100,255,0.5)'
        ), row=4, col=1)

        # Update layout
        fig.update_layout(
            title=f'{symbol} - Golden Path Analysis',
            xaxis_rangeslider_visible=False,
            height=800,
            template='plotly_dark'
        )

        return fig

    def generate_report(self, symbol):
        """Generate comprehensive analysis report"""
        if symbol not in self.data:
            return "No data available for analysis"

        data = self.data[symbol]
        latest = data.iloc[-1]

        # Market condition
        is_sideways = self.detect_sideways_market(symbol)
        whale_signals = self.analyze_whale_signals(symbol)
        entry_signals = self.generate_entry_signals(symbol)

        report = f"""
🚀 GOLDEN PATH ANALYSIS REPORT - {symbol}
{'='*50}

📊 CURRENT PRICE: ${latest['Close']:.2f}
📈 24h CHANGE: {((latest['Close'] - data.iloc[-2]['Close']) / data.iloc[-2]['Close'] * 100):.2f}%

🔍 TECHNICAL INDICATORS:
- RSI: {latest['RSI']:.1f} ({'Oversold' if latest['RSI'] < 30 else 'Overbought' if latest['RSI'] > 70 else 'Neutral'})
- Price vs EMA20: {((latest['Close'] - latest['EMA_20']) / latest['EMA_20'] * 100):.2f}%
- Bollinger Position: {((latest['Close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower']) * 100):.1f}%
- Volume Ratio: {latest['Volume_Ratio']:.2f}x average

🎯 MARKET CONDITION:
- Sideways Market: {'YES - Hidden Game Active' if is_sideways else 'NO - Trending Market'}

🐋 WHALE ACTIVITY (Last 10 signals):
"""
        for signal in whale_signals:
            report += f"- {signal['date'].strftime('%Y-%m-%d')}: {signal['type'].upper()} (Vol: {signal['volume_ratio']:.1f}x)
"

        report += f"""
⚡ TRADING SIGNALS:
"""
        for signal in entry_signals:
            report += f"- {signal}
"

        report += f"""
🛡️ RISK MANAGEMENT:
- Support Level: ${data['Low'].tail(20).min():.2f}
- Resistance Level: ${data['High'].tail(20).max():.2f}
- Stop Loss Suggestion: ${latest['Close'] * 0.95:.2f} (-5%)
- Position Size: Risk 1-3% of portfolio (Golden Path principle)

⭐ PSYCHOHISTORIAN'S NOTE:
This analysis supports building your alternative financial system.
Current market conditions {'favor' if len(entry_signals) > 0 else 'require caution for'}
accumulation strategies. Stay disciplined for humanity's future! 🌟
"""

        return report

# Example usage and testing
if __name__ == "__main__":
    print("🚀 Initializing Trade Trends Analyzer...")
    print("=" * 60)

    analyzer = TradeTrendsAnalyzer()

    # Analyze Bitcoin and Ethereum
    symbols = ["BTC", "ETH"]

    for symbol in symbols:
        print(f"\n📈 Analyzing {symbol}...")

        # Fetch and analyze data
        analyzer.fetch_crypto_data(symbol, period="6mo", interval="1d")
        analyzer.calculate_indicators(symbol)

        # Generate report
        report = analyzer.generate_report(symbol)
        print(report)

        # Create chart (optional - requires display)
        try:
            fig = analyzer.create_advanced_chart(symbol)
            if fig:
                fig.show()
        except:
            print(f"Chart display not available for {symbol} (run in Jupyter for charts)")

    print("\n✅ Analysis complete! Save this script as 'golden_path_analyzer.py'")
    print("📊 Run with: python golden_path_analyzer.py")
    print("💡 Tip: Install required packages with:")
    print("   pip install pandas numpy yfinance plotly talib")
