import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go

'''
May have to iterate through csv data to account for BirdCount containing distinct values at multiple time intervals
'''


data = pd.read_csv("Data/BirdData.csv")
print(data.head())

#Different variables should be tested for regression including weather, location, date, time, etc.
x = data["Weather"].values.reshape(-1, 1)
y = data["BirdCount"]

model = LinearRegression()
model.fit(x, y)

x_range = np.linspace(x.min(), x.max(), 100)
y_range = model.predict(x_range.reshape(-1, 1))


fig = px.scatter(data, x='TV', y='Sales', opacity=0.65)
fig.add_traces(go.Scatter(x=x_range, y=y_range, name='Linear Regression'))
fig.show()

# Note: In the above figure, the red line is a trend-line that shows the predictions made by
# the linear regression algorithm, which is nothing but the relationship between the weather
# at the time the video was taken and the number of birds present


# # you can have a look at all the predicted values
print(y_range)

# score will be good to compare with another models
linear_regression = LinearRegression()

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

linear_regression.fit(xtrain, ytrain)
accuracy = linear_regression.score(xtest, ytest)
print(accuracy)