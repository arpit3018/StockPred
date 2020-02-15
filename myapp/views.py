from django.shortcuts import render
import pandas as pd
import numpy as np
import datetime
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import math
import csv
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Lasso, BayesianRidge, Ridge
from sklearn.preprocessing import PolynomialFeatures
from .models import *
from newsapi import NewsApiClient
from django.http import HttpResponse

# Create your views here.
def index(request):
    res = Company.objects.all()
    return render(request,'home.html',{"res":res})

def csv_read_data(request):
    with open("./companylist.csv") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                company = Company(comp_name=row["Name"], industry=row["Sector"], symbol=row["Symbol"])
                company.save()

    with open("./ind_nifty500list.csv") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                sym = row["Symbol"] + ".NS"
                company = Company(comp_name=row["Name"], industry=row["Industry"], symbol=sym)
                company.save()
    return HttpResponse("Hello")

def get_result(request):
    if request.method == "POST":
        comp = request.POST.get("company")
        context_dict = predict_stock(comp)
        context_dict['name'] = comp
        flag = 0
        if "NS" in comp:
            flag = 1
        context_dict['comp_obj'] = Company.objects.filter(symbol=comp)[0]
        context_dict['flag'] = flag
        return render(request,'index.html',context_dict)
    return redirect("index")

def predict_stock(names):
    main_company = names
    start = datetime.datetime(2016, 10, 4)
    end = datetime.datetime(2020, 2, 15)

    df = web.DataReader(main_company, 'yahoo', start, end)

    high_value = df['High'][len(df['High'])-1]
    low_value = df['Low'][len(df['Low'])-1]
    open_value = df['Open'][len(df['Open'])-1]
    close_value = df['Close'][len(df['Close'])-1]

    close_px = df['Adj Close']
    mavg = close_px.rolling(window=100).mean()


    dfreg = df.loc[:,['Adj Close','Volume']]
    dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    dfreg.fillna(value=-99999, inplace=True)
    forecast_out = int(math.ceil(0.01 * len(dfreg)))
    forecast_col = 'Adj Close'
    dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
    X = np.array(dfreg.drop(['label'], 1))
    X = preprocessing.scale(X)
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]
    y = np.array(dfreg['label'])
    y = y[:-forecast_out]
    y_lately = y[-forecast_out:]

    X_train = X
    y_train = y

    clflasso = Lasso(alpha=0.1)
    clflasso.fit(X_train, y_train)

    X_test = X_lately
    y_test = y_lately

    confidence_lasso = clflasso.score(X_test,y_test)
    dfreg_lasso = dfreg.copy()

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
    plt.savefig('media/'+main_company+'.png')
    plt.clf()
    context_dict = {
    "res" : Company.objects.all(),
    "high_value" : round(high_value,2),
    "low_value" : round(low_value,2),
    "open_value" : round(open_value,2),
    "close_value" : round(close_value,2)
    }
    return context_dict


newsapi = NewsApiClient(api_key='a8e9c0ed7cc5444196beae086ae7abac')
def news(query):
    headlines = []
    top_headlines = newsapi.get_everything(q="Apple", language='en', page=4)
    articles = top_headlines["articles"]

    for article in articles:
        headlines.append(article["title"])
        return headlines
    return []