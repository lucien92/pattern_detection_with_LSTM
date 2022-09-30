#https://youtu.be/6S2v7G-OupA
"""
@author: Sreenivas Bhattiprolu
Shows errors on Tensorflow 1.4 and Keras 2.0.8
Works fine in Tensorflow: 2.2.0
    Keras: 2.4.3
dataset: https://finance.yahoo.com/quote/GE/history/
Also try S&P: https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC
"""

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Input, Dropout
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Model
import seaborn as sns

    
dataframe = pd.read_csv('/home/lucien/Documents/projet-data-science-pollution-master/donnees_txt_par_mois/2015_gamma_concat.txt')
df = dataframe[['Date', 'gamma']]
#df['Date'] = pd.to_datetime(df['Date'])

#on remarque qu'il y a un très gros pics vers 1633
# for val in df['gamma']:
#     #si la valeur dépasse 1000 afficher le numéro de la ligne
#     if val > 1000:
#         print(df.index[df['gamma'] == val].tolist())

sns.lineplot(x=df['Date'], y=df['gamma'])
plt.show()

print("Start date is: ", df['Date'].min())
print("End date is: ", df['Date'].max())


#Change train data from Mid 2017 to 2019.... seems to be a jump early 2017
train, test = df.loc[df['Date'] <= '130000'], df.loc[df['Date'] > '130000'] #on coupe à 130 000 pour avoir un des gros pics que l'on remarque sur le graphique


#Convert pandas dataframe to numpy array
#dataset = dataframe.values
#dataset = dataset.astype('float32') #COnvert values to float

#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
#scaler = MinMaxScaler() #Also try QuantileTransformer
scaler = StandardScaler()
scaler = scaler.fit(train[['gamma']])

train['gamma'] = scaler.transform(train[['gamma']])
test['gamma'] = scaler.transform(test[['gamma']])


#As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features. 
#In this example, the n_features is 2. We will make timesteps = 3. 
#With this, the resultant n_samples is 5 (as the input data has 9 rows).

seq_size = 300  # Number of time steps to look back (on choisit 300 en s'appuyant sur notre travail avec la matrix profile)
#Larger sequences (look further back) may improve forecasting.


def to_sequences(x, y, seq_size=1):
    x_values = []
    y_values = []

    for i in range(len(x)-seq_size):
        #print(i)
        x_values.append(x.iloc[i:(i+seq_size)].values)
        y_values.append(y.iloc[i+seq_size])
        
    return np.array(x_values), np.array(y_values)

trainX, trainY = to_sequences(train[['gamma']], train['gamma'], seq_size)
testX, testY = to_sequences(test[['gamma']], test['gamma'], seq_size)


# define Autoencoder model


#Input shape would be seq_size, 1 - 1 beacuse we have 1 feature. 
# seq_size = trainX.shape[1]

# model = Sequential()
# model.add(LSTM(128, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
# model.add(LSTM(64, activation='relu', return_sequences=False))
# model.add(RepeatVector(trainX.shape[1]))
# model.add(LSTM(64, activation='relu', return_sequences=True))
# model.add(LSTM(128, activation='relu', return_sequences=True))
# model.add(TimeDistributed(Dense(trainX.shape[2])))

# model.compile(optimizer='adam', loss='mse')
# model.summary()

#Try another model
model = Sequential()
model.add(LSTM(128, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dropout(rate=0.2))

model.add(RepeatVector(trainX.shape[1]))

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(rate=0.2))
model.add(TimeDistributed(Dense(trainX.shape[2])))
model.compile(optimizer='adam', loss='mae')
model.summary()

#on lance le training
# fit model
history = model.fit(trainX, trainY, epochs=2, batch_size=32, validation_split=0.1, verbose=1)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

#model.evaluate(testX, testY) #on évalue le training 

###########################
#Passons maintenant à la détection des anomalies sur notre propre signal, pour cela il est capital de définir un seuil de détection d'anomalie.

# trainPredict = model.predict(trainX)
# trainMAE = np.mean(np.abs(trainPredict - trainX), axis=1)
# plt.hist(trainMAE, bins=30) #donne  la moyenne des erreurs absolues calculée sur chaque channel de taille 300 qu'on a voulu prédire
# max_trainMAE = 0.3  #or Define 90% value of max as threshold. On l'a choisi à partir du graphique obtenu ci-dessus.

# testPredict = model.predict(testX)
# testMAE = np.mean(np.abs(testPredict - testX), axis=1)
# plt.hist(testMAE, bins=30)

# #Capture all details in a DataFrame for easy plotting
# anomaly_df = pd.DataFrame(test[seq_size:])
# anomaly_df['testMAE'] = testMAE
# anomaly_df['max_trainMAE'] = max_trainMAE
# anomaly_df['anomaly'] = anomaly_df['testMAE'] > anomaly_df['max_trainMAE']
# anomaly_df['gamma'] = test[seq_size:]['gamma']

# #Plot testMAE vs max_trainMAE
# sns.lineplot(x=anomaly_df['Date'], y=anomaly_df['testMAE'])
# sns.lineplot(x=anomaly_df['Date'], y=anomaly_df['max_trainMAE'])

# anomalies = anomaly_df.loc[anomaly_df['anomaly'] == True]

# #Plot anomalies
# sns.lineplot(x=anomaly_df['Date'], y=scaler.inverse_transform(anomaly_df['gamma']))
# sns.scatterplot(x=anomalies['Date'], y=scaler.inverse_transform(anomalies['gamma']), color='r')