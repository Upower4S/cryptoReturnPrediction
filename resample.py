import pandas as pd

def resample(df, resample_freq = '15min'):
    prices = pd.DataFrame()
    prices['close'] = df.close.resample(resample_freq).last()
    prices['open'] = df.open.resample(resample_freq).first()
    prices['high'] = df.high.resample(resample_freq).max()
    prices['low'] = df.low.resample(resample_freq).min()
    prices['volume'] = df.volume.resample(resample_freq).sum()
    prices['quote_asset_volume'] = df.quote_asset_volume.resample(resample_freq).sum()
    prices['taker_buy_base_volume'] = df.taker_buy_base_volume.resample(resample_freq).sum()
    prices['taker_buy_quote_volume'] = df.taker_buy_quote_volume.resample(resample_freq).sum()
    prices['number_of_trades'] = df.number_of_trades.resample(resample_freq).sum()
    return prices