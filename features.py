import pandas as pd
import talib
import numpy as np

def add_aroon_stochastic(
    df,
    high_col='high',
    low_col='low',
    aroon_length=14,
    stoch_length=14,
    smooth_d=3,
    prefix='aroon'
):  
    fe = pd.DataFrame()
    high = df[high_col].shift(1)
    low = df[low_col].shift(1)

    d_up = high.rolling(aroon_length).apply(
        lambda x: np.argmax(x[::-1]),  # 0 = most recent, aroon_length-1 = oldest
        raw=True
    )
    d_down = low.rolling(aroon_length).apply(
        lambda x: np.argmin(x[::-1]),
        raw=True
    )

    denom = (aroon_length - 1)
    df[f'{prefix}_up'] = 100 * (denom - d_up) / denom
    df[f'{prefix}_down'] = 100 * (denom - d_down) / denom

    df[f'{prefix}_osc'] = df[f'{prefix}_up'] - df[f'{prefix}_down']

    ao = df[f'{prefix}_osc']
    ao_min = ao.rolling(stoch_length).min()
    ao_max = ao.rolling(stoch_length).max()
    ao_range = ao_max - ao_min

    stoch_k = 100 * (ao - ao_min) / ao_range
    stoch_k = stoch_k.where(ao_range != 0, other=50)

    fe[f'{prefix}_stoch_k'] = stoch_k
    fe[f'{prefix}_stoch_d'] = stoch_k.rolling(smooth_d).mean()

    return fe

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

    h_l = df['high'].shift(1) - df['low'].shift(1)
    h_prev_close = (df['high'].shift(1)-df['close'].shift(2)).abs()
    l_prev_close = (df['low'].shift(1)-df['close'].shift(2)).abs()

    tr = pd.concat([h_l, h_prev_close, l_prev_close], axis = 1).max(axis = 1)
    X['ATR5'] = tr.rolling(5).mean()
    X['ATR10'] = tr.rolling(10).mean()

    X = X.join(add_aroon_stochastic(df))
    return X