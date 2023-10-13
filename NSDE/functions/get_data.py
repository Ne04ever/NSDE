import yfinance as yf
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import datetime
import math
from sklearn.preprocessing import StandardScaler


def load_data(stock, start_date='2005-01-01', end_date='2023-07-01'):
    stock_data = yf.download(stock, start=start_date, end=end_date)
    df = pd.DataFrame({'Adj_close': stock_data['Adj Close']})
    df['ret'] = df['Adj_close'].diff()
    df['log_close'] = np.log(df['Adj_close'])  
    df['log_ret'] = df['log_close'].diff()
    return df

