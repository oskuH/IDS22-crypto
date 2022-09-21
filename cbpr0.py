from tracemalloc import start
import cbpro
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#choose settings:
currency_pair = 'ETH-BTC'
start_date = '2022-06-01'
end_date = '2022-06-13'
gran = 3600

#fetch raw data:
c = cbpro.PublicClient()
rates = c.get_product_historic_rates(product_id=currency_pair, start=start_date, end=end_date, granularity=gran)

#create a dataframe with timestamp as index and with change of price added:
df = pd.DataFrame(rates, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
if df.empty == True:
    print("Too many data points for a single request!")
    print("Try again after editing dates and/or granularity.")
    exit()
df.index = pd.to_datetime(df['time'],unit='s')
df = df.drop(columns='time')
df = df.sort_values('time')
price = df['close'].array
pricechange = price[1:]-price[:-1]
pricechange = np.hstack((np.array([np.nan]),pricechange))
df['pricechange'] = pricechange

#reduced dataframe for initial work:
df_initial = df.loc[:,['volume','pricechange']]