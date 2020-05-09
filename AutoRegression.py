from datetime import date, timedelta
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
from pandas import Series
import numpy as np
import matplotlib.pyplot as plt
import random

def auto_reg(d1, d2):
    def difference(dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return np.array(diff)
    
    # invert difference value
    def inverse_difference(history, yhat, interval=1):
        return yhat + history[-interval]
    
    # load data set
    df = pd.read_csv('prsaf3.csv', index_col=False, header=0);
    series = df.transpose()[0]  # here we convert the DataFrame into a Serie
    # seasonal difference
    X = series.values
    days_in_year = 365
    differenced = difference(X, days_in_year)
    print("\n\n")
    print(differenced)
    differenced=differenced.astype(float) # argtype converts np.ndarray to the given type
    print(type(differenced))
    # fit model
    model = ARIMA(differenced, order=(7,0,1)) #differenced has to be in float
    model_fit = model.fit(disp=0)
    # multi-step out-of-sample forecast
    start_index = len(differenced)
    end_index = start_index + d2
    forecast = model_fit.predict(start=start_index, end=end_index)
    
    # invert the differenced forecast to something usable
    history = [x for x in X]
    day = 1
    i=0
    days=[[],[]]
    d=date.today()
    for yhat in forecast:
        inverted = inverse_difference(history, yhat, days_in_year)
        if (day>d1):
            days[0].append(str(d.day)+"-"+str(d.month)+"-"+str(d.year))
            days[1].append(inverted)
            d+=timedelta(days=1)
        print('Day %d: %f' % (day, inverted))
        history.append(inverted)
        day += 1
    #    tables=days
    #    tables=tables+value
    plt.plot(days[1], color='blue')
    count=random.randint(0,10)
    name="./static/auto-reg"+str(count)+".png"
    print(name)
    plt.savefig(name)
    plt.close()
    return [days, inverted,name]
