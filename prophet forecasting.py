# %%
from pandas import read_csv
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv'
df= read_csv(path,header=0)
df.head(5)

# %%
df.shape

# %%
#plot the dataframe
import matplotlib.pyplot as plt
df.plot(figsize=[12,8])

# %%
#df =df.set_index('Month')
df.plot(figsize=[12,8])
plt.xlabel("Month")
plt.ylabel("Sales")

# %%
#check the columns datatypes, time series requires data in a time data type
df.dtypes

# %%
import pandas as pd
#convert the month column to date time series data type
df['Month']= pd.to_datetime(df['Month'])
df['Month'].dtypes

# %%
# the prophet ()object takes arguments to configure the type of model you want, such as the type of growth adn seasonality
# the dataframe is then passed to tht fit () funtion
# fit () function takes the dataframe of time series data in this formats
# 1. the first columb must have the name 'ds' and contain date times
# 2. the second column must have the name 'y' and contain the observations
#change the column names
df.columns= ['ds','y']
df.head(2)

# %%
#!pip install prophet 

# %%
from prophet import Prophet


# %%
# define the model
model = Prophet()
# fit the model
model.fit(df)

# %%
#view the dataset tail 
df.tail(5)

# %% [markdown]
# ### In-sample-Forecast 

# %%
#define the period for which we want a prediction
f_list = list()
for i in range (1,13):
    date= '1968-%02d' % i
    f_list.append([date])
#create a dataframe with the future list
#future =pd.DataFrame[f_list]
#future.columns=['ds']
print(f_list)

# %%
df_list= pd.DataFrame(f_list)
df_list.head()

# %%
df_list.columns=['ds']

# %%
df_list.head()

# %%
# use the model to make a forecast
forecast = model.predict(df_list)
...
# summarize the forecast
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

# %%
# plot forecast
model.plot(forecast)
plt.show()

# %% [markdown]
# ### Out-of-Sample Forecast

# %%
#define the period beyond the end of the sample(training dataset )
fp_list=list()
for i in range(1,13):
    date='1969-%02d' % i
    fp_list.append([date])
print(fp_list)


# %%
#create a dataframe from the list
fp_list =pd.DataFrame(fp_list)
#view the dataframe
fp_list.head()



# %%
fp_list.columns = ['ds']


# %%
fp_list['ds']= pd.to_datetime(fp_list['ds'])
fp_list.dtypes

# %%
#forecast using the model
forecast_out= model.predict(fp_list)
plt.figure(figsize=[10,5])
plt.show()
model.plot(forecast_out)


# %% [markdown]
# ### Manually Evaluate Forecast Model

# %% [markdown]
#     This is to estimate the forecast model's performance 
#     Hold back some data from the model such as the last 12 months , fit the model on the first portion of the data using it to make predictions on the held-back portion and calculating an error measure ,such as the mean absolute error across the forecasts E.g A simulated out-of -Sample forecast

# %%
#create test dataset , remove the last 12 months
train_df= df.drop(df.index[-12:])
train_df.tail()

# %%
#A forecast can then be made on the last 12 months of date-times.
#define the model
model_train= Prophet()
#fit the model
model_train.fit(train_df)

# %%
#define the period for the prediction
train_list= list()
for i in range(1,13):
    date='1968-%02d' % i
    train_list.append([date])
train_list= pd.DataFrame(train_list)
train_list.columns= ['ds']
#use the model to make a forecast
forecast_train=model_train.predict(train_list)

# %%
import numpy as np
#define a MAe function
def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

# %%
#calculate MAE between expected and predicted values for december
y_true =df['y'][-12:].values
y_pred =forecast_train['yhat'].values
mae= mae(y_true,y_pred)
print('MAE: %.3f' % mae)

# %%
#plot the y_true and the y_pred
plt.plot(y_true,label='Actual')
plt.plot(y_pred,label='Predicted')
plt.legend()
plt.show()

# %%


# %%



