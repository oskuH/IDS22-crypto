from flask import Flask, request, render_template, json
import cbpro as cb
import pandas as pd
import numpy as np
import datetime as dt

from sklearn.metrics import classification_report, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
import statsmodels.api as sm

app = Flask(__name__)

class GetData:
    # """
    # Perform preprocessing on the selected cryptocurrency ticker
    # """
    def __init__(self, product_id, gra) -> None:
        self.__pub = cb.PublicClient() # initialize the coinbase client
        self.product_id = product_id
        self.gra = gra

    # get historic data      
    def historical_rates(self):
        "return historical data in float formats"
        historic_rates = pd.DataFrame(
                        self.__pub.get_product_historic_rates(product_id=self.product_id,granularity=self.gra),
                        columns=['time','low','high','open','close','volume'])
        
        return historic_rates.astype('float')

    def __repr__(self):
        return f"{self.historical_rates()}"

class Compute_statistics(GetData):

    def __init__(self, product_id, gra) -> None:
        # super().__init__(product_id, gra)

        self._historical_rates = GetData(product_id, gra).historical_rates()
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
        timeperiod = 14
        try:
            d = data.shape[1] # used to select the train data 
            data = data[::-1]
            index = data.columns
            new_data = pd.DataFrame()
            for i in range(d):
                new_data[index[i]] = data.iloc[:,i].ewm(span=timeperiod, adjust=False).mean()[timeperiod-1:]
            return new_data.dropna()[::-1]
        except IndexError:
            data = data[::-1]
            return data.ewm(span=timeperiod, adjust=False).mean()[timeperiod-1:][::-1]

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
        result = ys_test.iloc[1] - y_pred[0]
        prev_price = ys_test.iloc[1]

        s = np.sqrt(np.sum((model.predict(x_train)-y_train)**2)/(y_train.shape[0]-x_train.shape[1]-1))
        low, high = result - 1.96*s, result + 1.96*s

        return result, low, high

    def logit_reg(self):
        """Logistic regression prediction"""
        y = self._ychange.copy()
        y[y <= 0] = 0
        y[y > 0] = 1
        y = 1-y

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
        result = y_pred[0]
        real_data = y_test[0]

        return result,(y_pred[1:]==y_test[1:]).mean()

def fetch_pairs():
    c = cb.PublicClient()
    products_df = list(pd.DataFrame(c.get_products())['id'])
    products_df.sort()
    return products_df 

@app.route('/', methods=['GET','POST'])
def index():
    approved_pairs = ["BTC-EUR", "BTC-GBP", "BTC-USD", "BTC-USDC",
                    "BTC-USDT", "WBTC-USD", "CBETH-USD", "ETH-DAI",
                    "ETH-EUR", "ETH-GBP", "ETH-USD", "ETH-USDC",
                    "ETH-USDT", "MKR-USD"]

    if request.method == 'POST':
        if "different1" in request.form:
            input_pair = request.form.get('different1')
            input_time = request.form.get('different2')
            invalid_pair = False

            return render_template('home.html', 
                                    products=fetch_pairs(),
                                    aproducts = approved_pairs,
                                    ipair = input_pair,
                                    itime = input_time,
                                    bpair = json.dumps(invalid_pair))
        
        if "run" in request.form:
            invalid_pair = False
            input_pair = request.form.get('currency-pairs')
            input_time = request.form.get('times')
            if input_pair == "None":
                invalid_pair = True
                return render_template('home.html', 
                                        products = fetch_pairs(),
                                        aproducts = approved_pairs,
                                        ipair = input_pair,
                                        itime = input_time,
                                        bpair = json.dumps(invalid_pair))

            if input_time == "1 minute":
                addminutes = 1
                gran = 60
            elif input_time == "5 minutes":
                addminutes = 5
                gran = 300
            elif input_time == "15 minutes":
                addminutes = 15
                gran = 900
            elif input_time == "1 hour":
                addminutes = 60
                gran = 3600
            elif input_time == "6 hours":
                addminutes = 360
                gran = 21600
            elif input_time == "24 hours":
                addminutes = 1440
                gran = 86400

            utc = dt.datetime.now(dt.timezone.utc)
            utcplus = utc + dt.timedelta(minutes=addminutes)
            utc_times = []
            utc_times.append(f'{utc.day}/{utc.month}/{utc.year}')
            utc_times.append(utc.strftime('%H:%M:%S'))
            utc_times.append(utcplus.strftime('%H:%M:%S'))
            currencies = input_pair.split("-")

            model = Compute_statistics(product_id=input_pair,gra=gran)
            direction,accuracy = model.logit_reg()
            value, low, high = model.ln_reg()
            output =    [direction,'{:.2%}'.format(accuracy),'{:.2f}'.format(value),
                        '{:.2f}'.format(low), '{:.2f}'.format(high)]

            return render_template('results.html',
                                    utcs = utc_times,
                                    currency_list = currencies,
                                    output_list = output,
                                    pair = input_pair,
                                    time = input_time,
                                    add = addminutes,
                                    gra = gran)
        
        if "refresh1" in request.form:
            input_pair = request.form.get('refresh1')
            input_time = request.form.get('refresh2')
            addminutes = request.form.get('refresh3', type = int)
            gran = request.form.get('refresh4', type = int)

            utc = dt.datetime.now(dt.timezone.utc)
            utcplus = utc + dt.timedelta(minutes=addminutes)
            utc_times = []
            utc_times.append(f'{utc.day}/{utc.month}/{utc.year}')
            utc_times.append(utc.strftime('%H:%M:%S'))
            utc_times.append(utcplus.strftime('%H:%M:%S'))
            currencies = input_pair.split("-")

            model = Compute_statistics(product_id=input_pair,gra=gran)
            direction,accuracy = model.logit_reg()
            value, low, high = model.ln_reg()
            output =    [direction,'{:.2%}'.format(accuracy),'{:.2f}'.format(value),
                        '{:.2f}'.format(low), '{:.2f}'.format(high)]

            return render_template('results.html',
                                    utcs = utc_times,
                                    currency_list = currencies,
                                    output_list = output,
                                    pair = input_pair,
                                    time = input_time,
                                    add = addminutes,
                                    gra = gran)
    
    input_pair = json.dumps(None)
    input_time = json.dumps(None)
    invalid_pair = False

    return render_template('home.html', 
                            products=fetch_pairs(),
                            aproducts = approved_pairs,
                            ipair = input_pair,
                            itime = input_time,
                            bpair = json.dumps(invalid_pair))

if __name__ == "__main__":
    app.run(port=8080, debug=True)