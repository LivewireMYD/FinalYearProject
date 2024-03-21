#import of project

import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix 
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report
from sklearn import metrics
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from tkinter import *
from tkinter import font
from tkinter import Button




#visual tkiter init()
root=Tk()
root.geometry("500x500")
f=font.Font(weight="bold", family="Times New Roman", size=15)
frame=Frame(root,height="50px",width="500px",bg="red")
Label(frame,text="CRYPTOCURRENCY PRICE PREDICTION",foreground="white",bg="red",font=f).place(relx=0.15,rely=0.25)
frame.pack()
Label(root,text="some",font=f).pack()


#data processing
dataset = pd.read_csv("Data/BTC-USD.csv")
#print(dataset.shape)
#print(dataset.head())
#print((dataset['Close'] - dataset['Adj Close']).sum())
dataset["Date"] = pd.to_datetime(dataset['Date'])
#print(dataset["Date"])
dataset['Year'] = dataset['Date'].dt.year
#print(dataset)
#print(dataset.describe())
#print(dataset["Close"].shape)
dataset['pos_neg'] = dataset['Open'] - dataset['Close']
#if Open > Close then 0,else 1
dataset['Up/Down'] = np.where(dataset['pos_neg'] > 0, '0', '1')
#print(dataset['Up/Down'].unique())
#print(dataset.isna().sum())



#graph
def graph1():
    dataset['Close'].plot()
    plt.ylabel("Daily Bitcoin price")
    plt.show() 
def graph2():
    dataset['Close'].plot(style='k.',kind='hist')
    plt.title('Hisogram of closing price')
    plt.show() 
def graph3():
    by_year = dataset['Close'].groupby(dataset['Year']).mean()
    by_year.plot()
    plt.show() 
def graph4():
    plt.plot('Date', 'Volume', data=dataset)
    plt.xlabel('Year')
    plt.ylabel('Volume')
    plt.title('Volume trend')
    plt.show()
def graph5(): 
    ax1 = dataset.plot.scatter(x='Date', y='Close', c='DarkBlue')
    ax1 = dataset.plot.scatter(x='Date',y='Volume',c='DarkBlue')
    plt.show()


scaled_data = dataset[['Open', 'High', 'Low', 'Close', 'Volume']]
scaler = MinMaxScaler(copy=False)
scaled_data[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(scaled_data[['Open', 'High', 'Low', 'Close', 'Volume']])
print(scaled_data)


Label(root,text=scaled_data,font=f).pack()


scaled_data["Up/Down"] = dataset["Up/Down"].copy()

fig, axes = plt.subplots(1,2)
axes[0].scatter(dataset.Close, dataset.Close)
axes[0].set_title("Original data")
axes[1].scatter(scaled_data.Close, scaled_data.Close)
axes[1].set_title("MinMax scaled data")
#plt.show()

X = scaled_data[['Open', 'High', 'Low', 'Volume']]
y = scaled_data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



cv = KFold(n_splits=5, random_state=1, shuffle=True)
linReg = LinearRegression()
scores = cross_val_score(linReg, X, y, scoring='r2', cv=cv, n_jobs=1)
print(scores)

predicted = cross_val_predict(linReg, X, y, cv=6)
scores = cross_val_score(linReg, X, y, scoring='r2', cv=36, n_jobs=1)
print(scores)

fig,ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(1, 1, 1))
ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
ax.set_xlabel('Expected ')
ax.set_ylabel('Predicted ')
#plt.show()


fig,ax = plt.subplots()
plt.plot(scores)
ax.set_xlabel('Folds')
ax.set_ylabel('R2 score')
#plt.show()


kf = KFold(n_splits=10, shuffle=True)
scores = []
errors = []
i = 1

for training, testing in kf.split(X, y):
    X_train, X_test = X.loc[training], X.loc[testing]
    y_train, y_test = y.loc[training], y.loc[testing]
    print ('{}/10 Folds {}'.format(i, kf.n_splits))
    print('--------------------------')
    
    linReg = LinearRegression()
    linReg.fit(X_train, y_train)
    score = metrics.r2_score(y_test, linReg.predict(X_test))
    error = metrics.mean_squared_error(y_test, linReg.predict(X_test))
    print('R2 Score: ', score)
    print('MSE: ', error)
    scores.append(score)
    errors.append(error)
    print('\n')
    i += 1
fig, ax = plt.subplots()
plt.plot(errors)
ax.set_xlabel('Folds')
ax.set_ylabel('Errors')
#plt.show()

print("Coefficients: \n", linReg.coef_)
print("Intercept: \n", linReg.intercept_)

y_pred_lin = linReg.predict(X_test)
dfp = pd.DataFrame({'Actual_Price': y_test, 'Predicted_Price': y_pred_lin})
print(dfp.head())

y = linReg.coef_[0] * X_test['Open'] + linReg.coef_[1] * X_test['High'] + linReg.coef_[2] * X_test['Low'] + linReg.coef_[3] * X_test['Volume'] + linReg.intercept_

plt.subplots(1, figsize=(12,12))
plt.plot(np.linspace(0, y_test.size, y_test.size), y_test, color='red', label='Actual Data')
plt.plot(np.linspace(0, y_test.size, y_test.size), y, label='Predicted Data')
plt.grid(color='#000000', linestyle='-', linewidth=0.5)
plt.legend(loc="upper left")
##plt.show()

plt.scatter(dfp['Actual_Price'], dfp['Predicted_Price'])
#plt.show()

reg_score = linReg.score(X_test, y_test)
print("Linear Regression Score: ", reg_score)
print("Absolute Squared Error: ", mean_absolute_error(y_test, y_pred_lin))
mse_lin = mean_squared_error(y_test, y_pred_lin)
print("Mean Squared Error: ", mse_lin)


X = scaled_data[['Open', 'High', 'Low', 'Close','Volume']]
y = dataset["Up/Down"].values
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

cv = KFold(n_splits=5, random_state=1, shuffle=True)
lr = LogisticRegression()
scores = cross_val_score(lr, X, y, scoring='roc_auc', cv=cv, n_jobs=1)
print(scores)

predicted = cross_val_predict(lr, X, y, cv=6)
scores = cross_val_score(lr, X, y, scoring='roc_auc', cv=36, n_jobs=1)
print(scores)


fig,ax = plt.subplots()
plt.plot(scores)
ax.set_xlabel('Folds')
ax.set_ylabel('AUC Score')
#plt.show() 

print(dataset["Open"].sum())

linReg = LinearRegression()
linReg.fit(X, y)  # Train the model on the entire dataset
future_data = scaler.transform([[dataset["Open"].sum(), dataset["High"].sum(), dataset["Low"].sum(), dataset["Close"].sum(), dataset["Volume"].sum()]])
future_prediction = linReg.predict(future_data)
print(future_prediction[0]*-1)



Label(root,text="Future Predition : "+str(future_prediction[0]*-1),font=f).pack()
f=font.Font(weight="bold", family="Times New Roman", size=10)
frame=Frame(root,height="50px",width="500px",bg="red")
Button(frame,text="Graph 1",command=graph1,font=f).place(relx=.1,rely=.2)
Button(frame,text="Graph 2",command=graph2,font=f).place(relx=.23,rely=.2)
Button(frame,text="Graph 3",command=graph3,font=f).place(relx=.36,rely=.2)
Button(frame,text="Graph 4",command=graph4,font=f).place(relx=.49,rely=.2)
Button(frame,text="Graph 5",command=graph5,font=f).place(relx=.62,rely=.2)
frame.pack()

root.mainloop()