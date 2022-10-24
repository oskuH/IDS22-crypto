# make all the imports

import cbpro as cb
from cbpro.public_client import PublicClient
import numpy as np
import numpy.random as npr
import pandas as pd
import datetime as dt

from sklearn.metrics import classification_report, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
        # calculate change in variable values 
        self._xchange = self.__change_calculator(self._x)
        self._ychange = self.__change_calculator(self._y)
        # standardize output from variable change
        self._standardize_x = self.__standardizer(self._xchange)
        self._standardize_y = self.__standardizer(self._ychange)

        # split data set into test and training data
        self._final_x = self.__significant_vars()     

    def __change_calculator(self, data):
        
        data = data.values
        try:
            d = data.shape[1]
            return np.vstack([data[:-1,i] - data[1:,i] for i in range(d)]).T
        except IndexError:
            return (data[:-1] - data[1:])  
    
    def __standardizer(self, data):
        """
        standardize the data to avoid the use of intercept
        x is a numpy array
        d is an integer representing the dimension
        """
        #data = data.values
        try:
            d = data.shape[1]
            # return np.vstack([(data[:, i] - np.mean(data[:, i])) / (2*np.std(data[:, i])) for i in range(d)])
            for i in range(d):
                data[:, i] = (data[:, i] - np.mean(data[:, i])) / (2*np.std(data[:, i]))
            return data
        except IndexError:
            return (data - data.min()) / (data.max() - data.min())
    
    # correlation coefficient
    def __corrcoeffs(self):
        """Correlation coefficient of dataset"""
        d = self._standardize_x.shape[1]
        return [np.corrcoef(self._standardize_x[:,i], self._standardize_y)[0,1] for i in range(d)]


    # only variables with p-value less than 0.05
    def __significant_vars(self):
        '''Extract significant variables through permutation testing'''
        x = self._standardize_x
        x = pd.DataFrame(x, columns=['time','low','high','open','volume'])
        y = self._standardize_y

        # calculate pvalues
        logit_model = sm.Logit(y,x)
        result = logit_model.fit().pvalues

        # generate new index
        new_idx = result[result <= 0.05].index

        return x[new_idx]
    
    def ln_reg(self):
        """Linear regression prediction"""
        x_train, x_test, y_train, y_test =\
            train_test_split(self._final_x, self._standardize_y, test_size=0.2, random_state=100)

        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        report = r2_score(y_test, y_pred)

        return report

    def logit_reg(self):
        """Logistic regression prediction"""
        y = self._ychange
        y[y <= 0] = 0
        y[y > 0] = 1

        x_train, x_test, y_train, y_test =\
            train_test_split(self._final_x, y, test_size=0.2, random_state=100)

        model = LogisticRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        report = classification_report(y_test, y_pred)

        return report
    
    def __repr__(self):
        return f'''Logistic regression: {self.logit_reg()}
        Linear regression: {self.ln_reg()}'''

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

