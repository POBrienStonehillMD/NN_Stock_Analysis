import tensorflow as tf
import os, glob
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn import metrics

# Stock Ticker of file
ticker = "TSLA"

# NOTE: Path to save complete file
path = "C:/Users/Documents/"

# NOTE: Read data file and create it as a dataframe in pandas
df = pd.read_csv(path + ticker + "_2Year_technical_data.csv", na_values=['NA', ''])

z = df.drop(columns=['Unnamed: 0', 'date', 'time', 'open', 'high', 'low'], inplace=True)

list(df.columns)

x = df[['volume', 'SMA 60', 'EMA 60', 'vwap', 'MACD', 'MACD_Signal', 'MACD_Hist', 'SlowK', 'SlowD', 'RSI', 'AROON_Down', 'AROON_UP', 'ADX', 'CCI', 'MOM', 'AD', 'OBV', 'BBands_Upperband', 'BBands_Middleband', 'BBands_Lowerband', 'CDL3Inside', 'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY', 'CDLBREAKAWAY', 'CDLCOUNTERATTACK', 'CDLDOJI', 'CDLDOJISTAR', 'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN', 'CDLHIGHWAVE', 'CDLHOMINGPIGEON', 'CDLINVERTEDHAMMER', 'CDLKICKING', 'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP', 'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS', 'HT_TRENDLINE', 'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR_Inphase', 'HT_PHASOR_Quadrature', 'HT_Sine', 'HT_LeadSine', 'HT_TRENDMODE', 'ATR', 'NATR', 'TRANGE']].values
print(x)
y = df['close'].values
print(y)

# Build the neural network model
model = Sequential()

# Hidden layer 1
model.add(Dense(60, activation='relu'))

#Hidden Layer 2
model.add(Dense(40, activation='relu'))

#Hidden Layer 3
model.add(Dense(40, activation='relu'))

#Hidden Layer 4
model.add(Dense(20, activation='relu'))

#Hidden Layer 5 
model.add(Dense(20, activation='relu'))

#Hidden Layer 6
model.add(Dense(12, activation='relu'))

#Hidden Layer 7 
model.add(Dense(8, activation='relu'))

#Hidden Layer 8
model.add(Dense(8, activation='relu'))

# Output
model.add(Dense(1)) 

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x,y,verbose=1,epochs=400)

predictions = model.predict(x)
print(predictions)

# Measure RMSE error.  RMSE is common for regression.
score = np.sqrt(metrics.mean_squared_error(predictions,y))
print(f"Final score (RMSE): {score}")

# Sample predictions
for i in range(x.shape[0]):
    print(f"{i+1}. Actual Close: {y[i]} and the Predicted Close: {predictions[i][0]} with a Difference of: {y[i] - predictions[i][0]}")
