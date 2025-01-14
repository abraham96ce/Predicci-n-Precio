import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import time
import os
import sys
try:
  import ccxt
except:
  !pip install ccxt
  import ccxt

print(ccxt.exchanges) # print a list of all available exchange classes

msec = 1000
minute = 60 * msec
hold = 30

exchange = ccxt.bitso()  # <----- EXCHANGE al que se conecta para leer la información histórica

from_datetime = '2015-10-01 00:00:00'  #<--------------------- FECHA A partir de la cual se hace la lectura
from_timestamp = exchange.parse8601(from_datetime)

t = time.time()
now = exchange.milliseconds()
data = []

while from_timestamp < now:
    try:
       # print(exchange.milliseconds(), 'Fetching candles starting from', exchange.iso8601(from_timestamp))
        ohlcvs = exchange.fetch_ohlcv('BTC/USD', '1h', from_timestamp)   # <-------- criptomoneda
       # print(exchange.milliseconds(), 'Fetched', len(ohlcvs), 'candles')
        if len(ohlcvs) > 0:
          #  first = ohlcvs[0][0]
          #  last = ohlcvs[-1][0]
          #  print('First candle epoch', first, exchange.iso8601(first))
          #  print('Last candle epoch', last, exchange.iso8601(last))
            ## from_timestamp += len(ohlcvs) * minute * 5  # very bad
            from_timestamp = ohlcvs[-1][0] + minute * 60  # good
            #print(exchange.iso8601(from_timestamp))

            data += ohlcvs

    except (ccxt.ExchangeError, ccxt.AuthenticationError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as error:
        print('Got an error', type(error).__name__, error.args, ', retrying in', hold, 'seconds...')
        time.sleep(hold)

print(f'Tiempo de recuperación del historial {time.time() - t:.5f} sec')

dataset = pd.DataFrame(data, columns=['timestamp', 'open', 'high','low', 'close','volume'])

fecha = []
hora = []
for id in range(dataset.shape[0]):
  dataset.timestamp[id] = exchange.iso8601(int(dataset.timestamp[id]))
  aux1, aux2 = dataset.timestamp[id].split(sep='T')
  fecha.append(aux1)
  aux1, aux2 = aux2.split(sep='Z')
  hora.append(aux1)

#dataset['fecha'] = fecha
#dataset['hora'] = hora

primero = 0
numregistros = dataset.shape[0]
ultimo = dataset['timestamp'][numregistros-1]
#print(primero, ultimo)

x = dataset['timestamp']
y = dataset['high']

#print ("Display current time")
#current_time = datetime.now()
#print ("current time:-", current_time)
#print ("Display timestamp")
#time_stamp = current_time.timestamp()
#print ("timestamp:-", time_stamp)

dataset

#primero = dataset['fecha'][0]
numregistros = dataset.shape[0]
#ultimo = dataset['fecha'][numregistros-1]
#print(primero, ultimo)

x = dataset['timestamp']
y = dataset['high']

fig,ax = plt.subplots(figsize=(16,5))

#ax.plot(x, dataset['open'] , 'm-',  linewidth = 1.2, label = 'open' )
#ax.plot(x, dataset['close'],  'm.', label = 'close', markersize=1 )
ax.plot(x, dataset['high'],  'm-', label = 'high' )
#ax.plot(x, dataset['low'],   'c:', label = 'low ' )

plt.grid( color = 'green', linestyle = '--', linewidth = 0.2, axis = 'y' )
plt.title('Precio BTC/USDT\nBitso')
plt. tick_params (left        = False,
                  labelleft   = True,
                  bottom      = False,
                  labelbottom = False)
plt.ylabel("USD")



plt.xlabel('Histórico por hora')
plt.legend()
plt.show()
