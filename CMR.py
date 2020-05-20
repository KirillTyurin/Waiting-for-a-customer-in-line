import pandas as pd
import datetime as dt
import numpy as np
import matplotlib
import os

# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from tensorflow.python import keras


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i


def file_read(path, numberOfRows):
    indexCounter = 0
    with open(path, 'r') as file:
        nRows = numberOfRows
        nColumns = 4
        dataset = np.zeros(shape=(nRows, nColumns))
        arrivalTimes = []
        for line in file:
            try:
                dataInstance = line.split(';')
                arrivalTime = dataInstance[1]  # splits the line at the comma and takes the first bit
                try:
                    arrivalTime = dt.datetime.strptime(arrivalTime, '%H')
                    arrivalTime_hours = arrivalTime.hour
                    arrivalTime_minutes = 0
                except:
                    arrivalTime = dt.datetime.strptime(arrivalTime, '%H:%M')
                    arrivalTime_hours = arrivalTime.hour
                    arrivalTime_minutes = arrivalTime.minute

                arrivalHour = arrivalTime_hours
                arrivalMinute = arrivalTime_minutes
                waitingMinutes = dataInstance[2]
                serviceMinutes = dataInstance[3]

                arrivalTimes.append(arrivalTime)
                dataset[indexCounter] = [arrivalHour, arrivalMinute, waitingMinutes, serviceMinutes]
                indexCounter = indexCounter + 1
            except Exception:
                # print('index' + str(indexCounter) + 'error')
                pass
    return dataset, arrivalTimes, indexCounter


filenames = []
rootFilePath = '/Users/kirillturin/Desktop/Neural network/BankDataCsv/'
fullDataset = pd.DataFrame()

for bankCounter in range(3):
    for weekCounter in range(4):
        for dayCounter in range(5):
            filename = 'Bank' + str(bankCounter + 1) + 'Week' + str(weekCounter + 1) + 'Day' + str(dayCounter + 1)
            fullPath = rootFilePath + filename + '.csv'
            filenames.append(fullPath)

            numberOfRows = file_len(fullPath) - 1

            print('Reading ' + filename + 'that contains' + str(numberOfRows) + ' entries')
            tempFeatures, tempArrivalTimes, index_counters = file_read(rootFilePath + filename + '.csv', numberOfRows)
            dfTempFeatures = pd.DataFrame(np.array(tempFeatures),
                                          columns=['hour', 'minutes', 'waitingTime', 'serviceTime'])
            dfTempArrivalTimes = pd.DataFrame(np.array(tempArrivalTimes), columns=['arrivalTime'])
            numberOfRows = min(len(dfTempArrivalTimes), len(dfTempArrivalTimes))
            # numberOfRows = index_counters - 1
            timeLeavingTheQueue = []

            for arrivalTimeCounter in range(numberOfRows):
                k = dfTempArrivalTimes.at[arrivalTimeCounter, 'arrivalTime']
                e = pd.Timedelta(minutes=dfTempFeatures.at[arrivalTimeCounter, 'waitingTime'])
                timeLeavingTheQueue.append(k + e)
            dftimeLeavingTheQueue = pd.DataFrame(np.array(timeLeavingTheQueue), columns=['timeLeavingTheQueue'])

            waitingPeople = np.zeros(numberOfRows)
            for i in range(numberOfRows):
                for j in range(i):
                    if (dfTempArrivalTimes.at[i, 'arrivalTime'] < dftimeLeavingTheQueue.at[j, 'timeLeavingTheQueue']):
                        waitingPeople[i] += 1
            dfWaitingPeople = pd.DataFrame(np.array(waitingPeople), columns=['waitingPeople'])

            dayOfWeek = np.zeros(numberOfRows)
            for i in range(numberOfRows):
                dayOfWeek[i] = dayCounter
            dfDayOfWeek = pd.DataFrame(np.array(dayOfWeek), columns=['dayOfWeek'])

            dfWaitingPeople['waitingPeople'] = dfWaitingPeople['waitingPeople'].astype(int)
            dfTempFeatures['hour'] = dfTempFeatures['hour'].astype(int)
            dfTempFeatures['minutes'] = dfTempFeatures['minutes'].astype(int)
            dfDayOfWeek['dayOfWeek'] = dfDayOfWeek['dayOfWeek'].astype(int)

            tempDataset = pd.concat([dfTempFeatures, dfWaitingPeople, dfDayOfWeek], axis=1)
            tempDataset = tempDataset.reset_index(drop=True)
            fullDataset = pd.concat([fullDataset, tempDataset], axis=0)

fullDataset = fullDataset.reset_index(drop=True)
numberOfRows = fullDataset.size

print(fullDataset.shape[0])

print(f'Строк в датасете:{fullDataset.shape[0]}  столобцов: {fullDataset.shape[1]} columns.')

fullDataset.head()

fullDataset.describe()

print(f"Пропущенные значения {fullDataset.isnull().any().sum()}")

myMean = fullDataset["waitingTime"].mean()
print(f'Средннее время ожидания {myMean}')

myMedian = fullDataset["waitingTime"].median()
print(f'Медиана время ожидания {myMedian}.')

myStd = fullDataset["waitingTime"].std()
print(f'Стандартное откданение {myStd} в минутах.')

myMean2 = fullDataset["waitingPeople"].mean()
print(f'Среднее значение людей для ожидания {myMean2}')

myMedian2 = fullDataset["waitingPeople"].median()
print(f'Медиана людей ожидающих очереди  {myMean2}.')

myStd2 = fullDataset["waitingPeople"].std()
print(f'Стандартное отклонение ожидающих людей {myStd2} minutes.')

# data to be plotted
mu = fullDataset["waitingPeople"].mean()  # mean of distribution
sigma = fullDataset["waitingPeople"].std()  # standard deviation of distribution
x = fullDataset["waitingPeople"]

num_bins = 111

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(x, num_bins, density=1)

# add a 'best fit' line
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))

ax.plot(bins, y, '--')
ax.set_xlabel('Количестов людей в очереди')
ax.set_ylabel('Плотность распределения')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.savefig('Plots/waitingPeopleHistogram.pdf')

# data to be plotted
mu = fullDataset["waitingTime"].mean()  # mean of distribution
sigma = fullDataset["waitingTime"].std()  # standard deviation of distribution
x = fullDataset["waitingTime"]

num_bins = 33

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(x, num_bins, density=1)

# add a 'best fit' line
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
ax.plot(bins, y, '--')
ax.set_xlabel('Ожидание клиента (в минутах)')
ax.set_ylabel('Probability density')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.savefig('Plots/waitingTimeHistogram.pdf')


workingCopyDataset = fullDataset
workingCopyDataset.drop(['serviceTime'], axis=1);


def mean_encoder_regression(input_vector, output_vector):
    assert len(input_vector) == len(output_vector)
    numberOfRows = len(input_vector)

    temp = pd.concat([input_vector, output_vector], axis=1)
    # Compute target mean
    averages = temp.groupby(by=input_vector.name)[output_vector.name].agg(["mean", "count"])

    print (input_vector.isna().sum())
    print(averages)
    return_vector = pd.DataFrame(0, index=np.arange(numberOfRows), columns={'feature'})

    for i in range(numberOfRows):
        #return_vector.iloc[i] = averages['mean'][input_vector.iloc[i]]
        return_vector.iloc[i] = input_vector.iloc[i]

    return return_vector


workingCopyDataset.dropna(subset=['hour'], inplace=True)
workingCopyDataset.dropna(subset=['minutes'], inplace=True)
workingCopyDataset.dropna(subset=['dayOfWeek'], inplace=True)
workingCopyDataset.dropna(subset=['waitingTime'], inplace=True)

#encoded_input_vector_hour = workingCopyDataset['hour']
#encoded_input_vector_minutes = workingCopyDataset['minutes']
#encoded_input_vector_dayOfWeek = workingCopyDataset['dayOfWeek'];
encoded_input_vector_hour = mean_encoder_regression(workingCopyDataset['hour'], workingCopyDataset['waitingTime'])
encoded_input_vector_hour.columns = ['hour']
encoded_input_vector_minutes = mean_encoder_regression(workingCopyDataset['minutes'], workingCopyDataset['waitingTime'])
encoded_input_vector_minutes.columns = ['minutes']
encoded_input_vector_dayOfWeek = mean_encoder_regression(workingCopyDataset['dayOfWeek'], workingCopyDataset['waitingTime'])
encoded_input_vector_dayOfWeek.columns = ['dayOfWeek']

#workingCopyDataset.dropna(subset=['waitingPeople'], inplace=True)

'''
encoded_input_vector_hour['hour'] = encoded_input_vector_hour['hour'].reset_index(drop=True);
encoded_input_vector_minutes['minutes'] = encoded_input_vector_minutes['minutes'].reset_index(drop=True);
encoded_input_vector_dayOfWeek['dayOfWeek'] = encoded_input_vector_dayOfWeek['dayOfWeek'].reset_index(drop=True);
workingCopyDataset['waitingPeople'] = workingCopyDataset['waitingPeople'].reset_index(drop=True);
'''

'''
encoded_input_vector_hour['hour'] = workingCopyDataset['hour']
encoded_input_vector_minutes['minutes'] = workingCopyDataset['minutes']
encoded_input_vector_dayOfWeek['dayOfWeek'] = workingCopyDataset['dayOfWeek'];
workingCopyDataset['waitingPeople'] = workingCopyDataset['waitingPeople']
'''

X = pd.concat(
    [encoded_input_vector_hour['hour'].reset_index(drop=True), encoded_input_vector_minutes['minutes'].reset_index(drop=True), workingCopyDataset['waitingPeople'].reset_index(drop=True),
     encoded_input_vector_dayOfWeek['dayOfWeek'].reset_index(drop=True), workingCopyDataset['waitingTime']], axis = 1)

X.dropna(subset=['hour', 'minutes', 'waitingPeople', 'dayOfWeek', 'waitingTime'], inplace=True)

#y = workingCopyDataset['waitingTime'].head(len(X.index))

y = X['waitingTime'].values

X = X.drop(['waitingTime'], axis = 1)



print(X)

X.describe()

trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=42)
print(trainX.shape, trainy.shape)
print(testX.shape, testy.shape)


def scale_input(X, means, stds):
    #return (X - means) / stds
    return X


def descale_input(X, means, stds):
    #return (X * stds) + means
    return X

meansX = trainX.mean(axis=0)
stdsX = trainX.std(axis=0) + 1e-10



trainX_scaled = scale_input(trainX, meansX, stdsX)
testX_scaled = scale_input(testX, meansX, stdsX)

# create a deep learning model
inputVariables = 4
model = keras.models.Sequential()
model.add(keras.layers.Dense(12, input_dim=inputVariables, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='linear'))
model.summary()

model.compile(loss='mae', optimizer='adam')

# train model
numberOfEpochs = 500
batchSize = 256
history = model.fit(trainX_scaled, trainy, epochs=numberOfEpochs, batch_size=batchSize, verbose=1, validation_split=0.2)

print(history.history.keys())

# summarize history for loss
plt.plot(history.history['loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.savefig('loss.pdf')


plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Validation'], loc='upper left')
plt.savefig('lossValid.pdf')

testy_pred = model.predict(testX_scaled)
myLength = len(testy_pred)
plt.plot(range(myLength), testy)
plt.plot(range(myLength), testy_pred)
plt.ylabel('Customer waiting time (mins)')
plt.xlabel('Client')
plt.legend(['Real', 'Predicted'], loc='upper left')
plt.savefig('./plots/realVsPredictedWaitingTimes.pdf')


myMae = mean_absolute_error(testy, testy_pred)
print(f'Средняя ошибка {myMae}.')