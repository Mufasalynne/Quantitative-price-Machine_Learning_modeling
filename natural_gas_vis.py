import pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# opening the data file
data = pd.read_csv(r"C:\Users\Admin\Downloads\PYTHON_STUDY\J.P Morgan\Nat_Gas.csv")

# Analysing the data

# first five rows
print(data.head())

#last five rows
print(data.tail())

#check data information
print(data.info())

#check data type
print(data.dtypes)

#check missing values
print(data.isnull().sum())

# convert the Dates data from str to date time data type
data['Dates'] = pd.to_datetime(data['Dates'])

print(data.dtypes)

# start plotting the graph to a visual look of how the data looks like


plt.figure(figsize=(10,5))
plt.plot(data['Dates'],
         data['Prices'],
         marker='o',
         linestyle='-')

plt.title('Natural Gas Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.show()

# Build the interpolation Model

# set the date column as index to easily get the dates when searching for them

data.set_index('Dates', inplace=True)
print(data)

# set the dates data from monthly to daily dates
daily_data = data.asfreq("D")

print(daily_data.head(10))

# filling in the mising price data using the time interpolation

daily_data['Prices'] = daily_data['Prices'].interpolate (method="time")
print(daily_data)

#check for missing values in the daily data
print(daily_data.isnull().sum())


#plot the daily data
plt.figure(figsize=(10,5))
plt.plot(
         daily_data.index,
         daily_data['Prices'],
         #marker='o',
         linestyle='-' )

plt.title('Daily Prices for Natural Gas')
plt.xlabel('Dates')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.show()

#extrapolation for an extra year


# Build the forecasting model
model = ExponentialSmoothing(
    data['Prices'],
    trend='add',
    seasonal="add",
    seasonal_periods=12
)

#train the forecasting model
fitted_model = model.fit()

# forecast the next 12 months
forecast = fitted_model.forecast(12)

# cionvert the   forecast series into a dataset
forecast_df = forecast.to_frame(name='Prices')
print(forecast_df)

# plot historical and forecast graphs
plt.figure(figsize=(10,5))
plt.plot(
    data.index,
    data['Prices'],
    label = 'Historical Prices',
    marker ='o'
)

# visualize price forecast graph
plt.plot(
    forecast_df.index,
    forecast_df['Prices'],
    label = 'Forecast',
    marker ='o',
    linestyle='--'
)

plt.title('Historical and Forecast Natural Gas Prices ')
plt.xlabel('Date')
plt.ylabel('Prices')
plt.legend()
plt.grid(True)

plt.show()


# set the dates data from monthly to daily dates
daily_forecast = forecast_df.asfreq("D")

# interpolate the focused data 
daily_forecast['Prices'] = daily_forecast['Prices'].interpolate(method='time')
print(daily_forecast)

# the script should take a date as input and return a price estimate
# combine the historical and the forecast data
combined_data = pd.concat([daily_data, daily_forecast])
print(combined_data.tail(20))

#Build a function that take a date as input and return a price estimate
def estimate_price(input_date):
    input_date = pd.to_datetime(input_date)

    # check if the input date is with in the combined data
    if input_date < combined_data.index.min() or input_date > combined_data.index.max():
        return "Date is outside the available range." 
    
        # monthly forecast
    return combined_data.loc[input_date, "Prices"]
    
print(estimate_price("2023-03-15"))