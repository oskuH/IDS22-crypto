# make all the imports

import cbpro as cb
from cbpro.public_client import PublicClient
from matplotlib.streamplot import InvalidIndexError
import numpy as np
import numpy.random as npr
import pandas as pd
import datetime as dt

from sklearn.metrics import classification_report, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

import talib as ta
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm

class GetData:
    # """
    # Perform preprocessing on the selected cryptocurrency ticker
    # """
    def __init__(self, product_id) -> None:
        """
        product_id(str): The cryptocurrency ticker e.g. BTC-EUR
        before(datetime or str): Start time for data e.g. '2022-10-5 12:00:00' 
        after(datetime or str): End time for data e.g. datetime.datetime.now()
        """
        self.__pub = PublicClient() # initialize the coinbase client
        self.product_id = product_id


    # get historic data      
    def historical_rates(self):
        "return historical data in float formats"
        historic_rates = pd.DataFrame(
                        self.__pub.get_product_historic_rates(product_id=self.product_id),
                        columns=['time','low','high','open','close','volume'])
        
        return historic_rates.astype('float')


    def __repr__(self):
        return f"{self.historical_rates()}"

class Compute_statistics(GetData):
    """
    computes the statistics of the data
    """

    def __init__(self, product_id) -> None:
        super().__init__(product_id)

        self._historical_rates = GetData('BTC-EUR').historical_rates()
        self._x = self._historical_rates.drop(columns=['close'])
        self._y = self._historical_rates.close
        # smooth data
        self._smooth_x = self.smooth_data(self._x)
        self._smooth_y = self.smooth_data(self._y)
        # calculate change in variable values 
        self._xchange = self.__change_calculator(self._smooth_x)
        self._ychange = self.__change_calculator(self._smooth_y)
        # standardize output from variable change
        self._standardize_x = self.__standardizer(self._xchange)
        self._standardize_y = self.__standardizer(self._ychange)

        # # split data set into test and training data
        self._final_x = self.__significant_vars()     

    def smooth_data(self, data):
        data = data.copy()
        try:
            d = data.shape[1] # used to select the train data 
            data = data[::-1]
            index = data.columns
            new_data = pd.DataFrame()
            for i in range(d):
                new_data[index[i]] = ta.EMA(data.iloc[:,i], timeperiod=14)
            return new_data.dropna()[::-1]
        except IndexError:
            data = data[::-1]
            return ta.EMA(data, timeperiod=14).dropna()[::-1]

    def __change_calculator(self, data):
        """data is a pandas df."""
        data = data.copy()
        try:
            d = data.shape[1] # used to select the train data 
            return data[1:].diff(periods=1).dropna().reset_index(drop=True)
        except IndexError:
            return data[:-1].diff(periods=1).dropna().reset_index(drop=True)

    
    def __standardizer(self, data):
        """
        standardize the data to avoid the use of intercept
        x is a pandas data frame
        """
        x = data.copy()
        try:
            d = x.shape[1]
            x = x.sub(x.mean(axis=0), axis='columns')
            x = x.div(2*x.std(axis=0), axis='columns')
            return x
        except IndexError:
            return (x - x.min()) / (x.max() - x.min())
    
    # correlation coefficient
    def __corrcoeffs(self):
        """Correlation coefficient of dataset"""
        d = self._standardize_x.shape[1]
        return [np.corrcoef(self._standardize_x[:,i], self._standardize_y)[0,1] for i in range(d)]


    # only variables with p-value less than 0.05
    def __significant_vars(self):
        '''Extract significant variables through permutation testing'''

        x = self._standardize_x.copy()
        y = self._standardize_y.copy()

        # calculate pvalues
        logit_model = sm.Logit(y,x)
        result = logit_model.fit()
        pvalues = result.pvalues
        idx = pvalues[pvalues <= 0.5].index

        return self._xchange[idx]
    
    def ln_reg(self):
        """Linear regression prediction"""
        x_train, x_test, y_train, y_test =\
            train_test_split(self._final_x[::-1], self._ychange[::-1], test_size=0.2, shuffle=False)    
        x_train, x_test, y_train, y_test = x_train[::-1], x_test[::-1], y_train[::-1], y_test[::-1]

        xs_train, xs_test, ys_train, ys_test =\
            train_test_split(self._smooth_x.iloc[:-1][::-1], self._smooth_y.iloc[:-1][::-1], test_size=0.2, shuffle=False)
        xs_train, xs_test, ys_train, ys_test = xs_train[::-1], xs_test[::-1], ys_train[::-1], ys_test[::-1]

        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        report = r2_score(y_test, y_pred)
        result = ys_test.iloc[1] + y_pred[0]
        real_data = ys_test.iloc[0]

        #return type(y_test)
        return result, real_data
        return report

    def logit_reg(self):
        """Logistic regression prediction"""
        y = self._ychange
        y[y <= 0] = 0
        y[y > 0] = 1

        x_train, x_test, y_train, y_test =\
            train_test_split(self._final_x[::-1], y[::-1], test_size=0.2, shuffle=False)    
        x_train, x_test, y_train, y_test = x_train[::-1], x_test[::-1], y_train[::-1], y_test[::-1]

        xs_train, xs_test, ys_train, ys_test =\
            train_test_split(self._smooth_x.iloc[:-1][::-1], self._smooth_y.iloc[:-1][::-1], test_size=0.2, shuffle=False)
        xs_train, xs_test, ys_train, ys_test = xs_train[::-1], xs_test[::-1], ys_train[::-1], ys_test[::-1]

        model = LogisticRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        report = classification_report(y_test, y_pred)
        result = y_pred[:10]
        real_data = y_test[:10]

        #return report
        return result, real_data
    
    def __repr__(self):
        # return f"{self._smooth_y}"
        #return f"{self._standardize_y}, {self._standardize_x}"
        # return f"{self.__significant_vars()}"
        #return f"{self._xchange}, {self._ychange}"
        
        #return f"{self._x['time']}, \n {self._xchange.time}"
        # return f'''Linear regression: {self.ln_reg()}'''
        return f'''Logistic regression: {self.logit_reg()}'''

class ApplicationInterface:
    # OSKARI'S SECTION !
    # call currency pair and make predictions
    def visualization(self):
        data = self.compute_stats()
        plt.scatter(data.time_change, data.price_change)
        plt.show()
        pass


if __name__ == '__main__':
    # test the predictor class
    data = GetData(product_id='BTC-EUR')
    #print(data)
    stats = Compute_statistics(product_id='BTC-EUR')
    print(stats)

