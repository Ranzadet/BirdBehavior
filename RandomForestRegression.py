import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

'''
May have to iterate through csv data to account for BirdCount containing distinct values at multiple time intervals
'''

data = pd.read_csv("Data/BirdData.csv")
print(data.head())

#Different variables should be tested for regression including weather, location, date, time, etc.
x = data["Weather"].values.reshape(-1, 1)
y = data["BirdCount"]

# Method-2: Random Forest regression algorithm
# the Random Forest regression algorithm to train
# the electricity price prediction model:

model = RandomForestRegressor()

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model.fit(xtrain, ytrain)

x_range = np.linspace(x.min(), x.max(), 100)
y_range = model.predict(x_range.reshape(-1, 1))

fig = px.scatter(data, x='TV', y='Sales', opacity=0.65)

fig.add_traces(go.Scatter(x=x_range, y=y_range,

                          name='Random Forest Regression'))

fig.show()