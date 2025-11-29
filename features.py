import pandas as pd
import talib
import numpy as np

def construct_features(df):
    X = pd.DataFrame()
        
    for col in ['open', 'high', 'low', 'close','number_of_trades', 'quote_asset_volume','taker_buy_quote_volume','quote_asset_volume',  'volume' ]:
        X[f'{col}_lag_1'] = df[col].shift(1)

    X['HminusL'] = X['high_lag_1'] - X['low_lag_1']
    
    X['return'] = np.log(df['close']) - np.log(X['close_lag_1'])

    for i in range(1,6):
        X[f'return_lag_{i}'] = X['return'].shift(i)

    X['SMA'] = talib.SMA(df['close'].shift(1), timeperiod=5)
    

    X['correlation_sma_close'] = X['SMA'].rolling(window=30, min_periods=30).corr(df['close'].shift(1))

    X['sum3'] = X[[f'return_lag_{i}' for i in range(1,4)]].sum(axis = 1)
    X['sum5'] = X[[f'return_lag_{i}' for i in range(1,6)]].sum(axis = 1)
    X['sum3-sum5'] = X['sum3'] - X['sum5']

    X['target']  = (X['return'] > 0).astype(int)

    del X['return']

    diff_price = df['close'].shift(1) - df['close'].shift(2)
    up = diff_price.where(diff_price>0, other= 0)
    down = - diff_price.where(diff_price<0, other = 0)
    SmmaU9  = talib.EMA(up,   9)
    SmmaD9  = talib.EMA(down, 9)
    SmmaU14 = talib.EMA(up,   14)
    SmmaD14 = talib.EMA(down, 14)

    def rsi_from_ud(u, d):
        denom = np.where(d == 0, np.nan, d)
        rs = u / denom
        rsi = 100.0 - 100.0 / (1.0 + rs)
        return np.where(np.isnan(rsi), 100.0, rsi)
    rsi9  = rsi_from_ud(SmmaU9,  SmmaD9)
    rsi14 = rsi_from_ud(SmmaU14, SmmaD14)
    X[f'RSI9']  = rsi9
    X[f'RSI14'] = rsi14
    
    X[f'RSI9Smaller20'] = (X[f'RSI9'] < 20).astype(int)
    X[f'RSI14Smaller20'] = (X[f'RSI14'] < 20).astype(int)
    X[f'RSI9Bigger80'] = (X[f'RSI9'] > 80).astype(int)
    X[f'RSI9Bigger80'] = (X[f'RSI14'] > 80).astype(int) 

    X['MACD1'],X['MACD2'],X['MACD3']  = talib.MACD(df['close'].shift(1), fastperiod=5, slowperiod=10, signalperiod=5)

    X['roc9'] = (df['close'].shift(1) - df['close'].shift(9))/df['close'].shift(9)
    X['roc14'] = (df['close'].shift(1) - df['close'].shift(14))/df['close'].shift(14)

    X['ewa'] = df['close'].shift(1).ewm(alpha=0.9).mean()

    X['mom5'] = talib.MOM(df['close'].shift(1), timeperiod=5)

    hhv = df['high'].shift(1).rolling(14).max()
    llv = df['low'].shift(1).rolling(14).min()
    denom = hhv - llv
    X['WilliamR'] = (hhv - df['close'].shift(1)) / denom.replace(0, np.nan)
    X['doubleEMA'] = talib.DEMA(df['close'].shift(1), timeperiod=10)
    return X