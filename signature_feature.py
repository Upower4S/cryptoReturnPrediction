import pandas as pd
import talib
import numpy as np
import iisignature
from tqdm import tqdm  # For progress bar

def create_signature_features(df, window_size=100, level=2):
    """
    Returns the signature of the cumulative log returns of a given time series of prices.
    
    Args:
        df: DataFrame with a 'close' column
        window_size: Lookback period (e.g., 20 days)
        level: Truncation level of the signature
        
    Returns:
        DataFrame containing signature features
    """
    log_prices = np.log(df['close'].values)
    sig_dim = iisignature.siglength(2, level) 
    features = np.full((len(df), sig_dim), np.nan) # Initialize with NaNs (because first 20 rows won't have a signature)
    t_vec = np.linspace(0, 1, window_size) # Normalized time vector
    
    print(f"Computing Rolling Signatures (Window: {window_size}, Level: {level})...")
    
    for i in tqdm(range(window_size, len(df))):
        segment_log_prices = log_prices[i - window_size : i]
        segment_cum_ret = segment_log_prices - segment_log_prices[0]
        path = np.column_stack([t_vec, segment_cum_ret])
        features[i] = iisignature.sig(path, level)
    
    col_names = [f'Sig_{k}' for k in range(sig_dim)]
    feat_df = pd.DataFrame(features, index=df.index, columns=col_names)
    
    return feat_df