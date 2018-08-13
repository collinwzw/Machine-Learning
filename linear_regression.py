import pandas as pd
import quandl
import math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

quandl.ApiConfig.api_key="CMouCxrjr8BiByjfHLHg"
df=quandl.get("SSE/PAT", authtoken="CMouCxrjr8BiByjfHLHg")

df = df[['High','Low','Last','Previous Day Price','Volume',]]
df['HL_PCT'] = (df['High'] - df['Last']) / df['Last'] * 100
df['PCT_change'] = (df['Last'] - df['Previous Day Price']) / df['Previous Day Price'] * 100

df = df[['Last', 'HL_PCT', 'PCT_change','Volume']]
#print(df.head())

forecast_col = 'Last'
df.fillna(-99999,inplace=True)

forecast_out = int (math.ceil(0.1*len(df)))
print(forecast_out)

df['label'] = df [forecast_col].shift(-forecast_out)

#print(df.head())

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

#clf = LinearRegression()
clf = svm.SVR()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy,forecast_out )

df['Forecast'] = np.nan
last_date=df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Last'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

