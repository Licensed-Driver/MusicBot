import architecture as arch
import torch
from keras.api.datasets import mnist
from functools import partial
from numbers import Number
import datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import *
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

API_KEY = "PKDBTTYGOCV0W5JMFD3A"
API_SECRET = "2dAcwatlZWVSd7zPlqHxZjvToVClJybOa7BvAlWc"
APCA_API_BASE_URL = "https://paper-api.alpaca.markets"

# Define constant parameters
start=datetime.datetime.now()-datetime.timedelta(days=5),
end=datetime.datetime.now()
tickers = ["NVDA", "AMD", "INTC", "SMCI"]
features = ["open", "high", "low", "close", "volume"]
timeframe = TimeFrame.Minute
lookback_days = 2 # Looking 2 days back for data
sequence_len = 60  # 60-minute sequences

data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

ACCOUNT = trading_client.get_account()

# Fetch Data
all_dfs = {}

for ticker in tickers:
    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=timeframe,
        start=start,
        end=end
    )
    bars = data_client.get_stock_bars(request)
    df = bars.df.reset_index()
    df = df[df['symbol'] == ticker]  # Make sure we only get this ticker
    df = df.sort_values('timestamp').reset_index(drop=True)
    all_dfs[ticker] = df

# Make sure each df is the same length
min_len = min(len(df) for df in all_dfs.values())
for ticker in tickers:
    all_dfs[ticker] = all_dfs[ticker].iloc[:min_len]

# Make dfs into a tensor
batches=[]
for i in range(min_len - sequence_len):
    batch = []
    for ticker in tickers:
        segment = all_dfs[ticker].iloc[i:i+sequence_len][features].values.T
        batch.append(segment)

    batch_tensor = torch.tensor(batch, dtype=torch.float32) # Tensor of shape [Tickers, Features, Time]
    batches.append(batch_tensor)

X = torch.stack(batches)

# Define The stock network

stock_net = arch.NN(X.shape, dtype=torch.float32)

stock_net.gen_con(out_channels=5, kernel_size=4, activation_func=arch.Functional.leaky, activation_deriv_func=arch.Functional.leakyDeriv, normalizer=arch.Functional.Batch_Normalizer)
stock_net.gen_fc(out_channels=2, activation_func=arch.Functional.softmax, activation_deriv_func=arch.Functional.linearDeriv, normalizer=arch.Functional.Batch_Normalizer)

loss_deriv = arch.Functional.softmax_cross_entropy_deriv

stock_net.train(X, )

# preparing market order
market_order_data = MarketOrderRequest(
                    symbol="SPY",
                    qty=0.023,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                    )

# Market order
market_order = trading_client.submit_order(
                order_data=market_order_data
               )

# preparing limit order
limit_order_data = LimitOrderRequest(
                    symbol="BTC/USD",
                    limit_price=17000,
                    notional=4000,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.FOK
                   )

# Limit order
limit_order = trading_client.submit_order(
                order_data=limit_order_data
              )

CASH = 1000
PORTFOLIO_VALUE = CASH