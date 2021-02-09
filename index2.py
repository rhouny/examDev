from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return (render_template('index2.html', url='new_plot.png'))


@app.route('/', methods=['POST'])
def result():
    data1 = request.form['data1']

    print(data1)
    df = pd.read_csv(data1)
    df = df.dropna()
    print(df.head())
    print(df.tail())
    df_close = df['Close']
    print(df_close.shape)
    plt.plot(df_close)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_close = scaler.fit_transform(np.array(df_close).reshape(-1, 1))
    print(df_close.shape)
    print(df_close)
    # Split the data into train and test split
    training_size = int(len(df_close) * 0.75)
    test_size = len(df_close) - training_size
    train_data, test_data = df_close[0:training_size, :], df_close[training_size:len(df_close), :1]

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 10
    x_train, y_train = create_dataset(train_data, time_step)
    x_test, y_test = create_dataset(test_data, time_step)
    # Reshape the input to be [samples, time steps, features] which is the requirement of LSTM
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    # Create the LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(10, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=64, verbose=1)
    # Lets predict and check performance metrics
    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)
    # Transform back to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # Calculate RMSE performance metrics
    math.sqrt(mean_squared_error(y_train, train_predict))
    # Test Data RMSE
    math.sqrt(mean_squared_error(y_test, test_predict))
    # Plotting

    # Shift train prediction for plotting
    look_back = 10
    trainPredictPlot = np.empty_like(df_close)
    trainPredictPlot[:, :] = 0
    trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

    # Shift test prediction for plotting
    testPredictPlot = np.empty_like(df_close)
    testPredictPlot[:, :] = 0
    testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df_close) - 1, :] = test_predict

    # Plot baseline and predictions
    plt.plot(scaler.inverse_transform(df_close))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    #plt.show()
    real_stock_price = scaler.inverse_transform(df_close).flatten()
    real_stock_price = real_stock_price.tolist()
    print('pretr',len(train_predict))
    print('prete', len(test_predict))
    e=len(train_predict)+10
    train_stock_price = trainPredictPlot.flatten()
    train_stock_price = train_stock_price.tolist()
    train_stock_price = train_stock_price[:e]
    a = 0
    for i in range(0, len(train_stock_price)):
        a = a + 1
    test_stock_price = testPredictPlot.flatten()
    test_stock_price = test_stock_price.tolist()
    test_stock_price = test_stock_price[a:]

    final=train_stock_price+test_stock_price
    #train_stock_price = train_stock_price[:len(train_stock_price) - 30]
    print(train_stock_price)

    print(test_stock_price)
    app = []
    for i in range(0, len(scaler.inverse_transform(df_close))):
        app.append(i)
    print(app)
    app = []
    for i in range(0, len(scaler.inverse_transform(df_close))):
        app.append(i)
    print(app)
    a = 0
    for i in range(0, len(train_stock_price)):
        a = a + 1
    print('a:',a)
    b = 0
    for i in range(0, len(test_stock_price)):
        b = b + 1
    print('b:',b)
    print('app:',app)
    return render_template('index2.html', url='/static/images/new_plot.png',
                           predicted_stock_price=final, real_stock_price=real_stock_price, app=app)


if __name__ == '__main__':
    app.run(debug=True)
