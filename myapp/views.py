from django.shortcuts import render
import pandas as pd
import numpy as np
import datetime
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import math
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Lasso, BayesianRidge, Ridge
from sklearn.preprocessing import PolynomialFeatures

# Create your views here.
def index(request):
    main_company = 'TATAMOTORS.NS'
    # Compare with similar company
    # Define the date range
    start = datetime.datetime(2016, 10, 4)
    end = datetime.datetime(2020, 2, 14)

    df = web.DataReader(main_company, 'yahoo', start, end)

    high_value = df['High'][len(df['High'])-1]
    low_value = df['Low'][len(df['Low'])-1]
    open_value = df['Open'][len(df['Open'])-1]
    close_value = df['Close'][len(df['Close'])-1]
    # print(df.tail())

    # Calculating the rolling mean for observation
    close_px = df['Adj Close']
    mavg = close_px.rolling(window=100).mean()


    dfreg = df.loc[:,['Adj Close','Volume']]
    dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    # Drop missing value
    dfreg.fillna(value=-99999, inplace=True)
    # We want to separate 1 percent of the data to forecast
    forecast_out = int(math.ceil(0.01 * len(dfreg)))
    # Separating the label here, we want to predict the AdjClose
    forecast_col = 'Adj Close'
    dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
    X = np.array(dfreg.drop(['label'], 1))
    # Scale the X so that everyone can have the same distribution for linear regression
    X = preprocessing.scale(X)
    # Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    # Separate label and identify it as y
    y = np.array(dfreg['label'])
    y = y[:-forecast_out]
    y_lately = y[-forecast_out:]



    X_train = X
    y_train = y

    # # 1. Linear regression 
    # clfreg = LinearRegression(n_jobs=-1)
    # clfreg.fit(X_train, y_train)

    # 2. Lasso - Linear Model
    clflasso = Lasso(alpha=0.1)
    clflasso.fit(X_train, y_train)

    # # 3. Bayesian Ridge - Linear Model
    # clfbr = BayesianRidge()
    # clfbr.fit(X_train, y_train)

    # # 4 Ridge - Linear Model
    # clfridge = Ridge(alpha=1.0)
    # clfridge.fit(X_train, y_train)

    X_test = X_lately
    y_test = y_lately

    # confidence_reg = clfreg.score(X_test, y_test)
    confidence_lasso = clflasso.score(X_test,y_test)
    # confidence_br = clfbr.score(X_test,y_test)
    # confidence_ridge = clfridge.score(X_test, y_test)

    # dfreg_lr = dfreg.copy()
    dfreg_lasso = dfreg.copy()
    # dfreg_ridge = dfreg.copy()
    # dfreg_br = dfreg.copy()

    noOfDaysData = 100

    forecast_set_lasso = clflasso.predict(X_lately)
    dfreg_lasso['Forecast_lasso'] =  np.nan

    last_date = dfreg_lasso.iloc[-1].name
    last_unix = last_date
    next_unix = last_unix + datetime.timedelta(days=0)

    for i in forecast_set_lasso:
        next_date = next_unix
        next_unix += datetime.timedelta(days=1)
        dfreg_lasso.loc[next_date] = [np.nan for _ in range(len(dfreg_lasso.columns)-1)]+[i]
    dfreg_lasso['Adj Close'].tail(noOfDaysData).plot()
    dfreg_lasso['Forecast_lasso'].tail(noOfDaysData).plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.savefig('media/TCS.png')
    context_dict = {
    "high_value" : round(high_value,2),
    "low_value" : round(low_value,2),
    "open_value" : round(open_value,2),
    "close_value" : round(close_value,2)
    }
    return render(request,'index.html',context_dict)

