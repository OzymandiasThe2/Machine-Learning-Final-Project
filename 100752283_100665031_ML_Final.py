"""
Shriji Shah (100665031)
Zachary Silver (100752283)
INFR 3700 Final Assignment
"""

# Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


data = pd.read_csv(
    'https://raw.githubusercontent.com/OzymandiasThe2/machine_learning_final_project/main/anime.csv')

# understanding the data
data.head()

# will show all the integer object collumns --> drop this shit
data.describe()

data.shape

# check for columns and rows

data.columns

# checking for unique values

data.nunique()

# cleaning up the data
## Confirm if any null columns
data.isnull().sum()

# dropping values

dataSet = data.drop(columns=["MAL_ID", "English name", "Japanese name", "Aired", "Premiered",
                             "Producers", "Licensors", "Duration", "Members", "Favorites", "Watching",
                             "Completed", "On-Hold", "Dropped", "Plan to Watch", "Genres",
                             "Score-10", "Score-9", "Score-8", "Score-7", "Score-6", "Score-5",
                             "Score-4", "Score-3", "Score-2", "Score-1"])

dataSet.head()

# shows only Popularity as ints, therefore we have to turn the rest of the int columns into actula intgers
dataSet.describe()

# drop the "unknown" values from the Score column
drop_score_unknown = dataSet[(dataSet["Score"] == "Unknown")].index
dataSet.drop(drop_score_unknown, inplace=True)

# convert the column from string to integers
dataSet["Score"] = dataSet["Score"].astype(str).astype(float)

# drop the "unknown" values from the Ranked column
drop_ranked_unknown = dataSet[(dataSet["Ranked"] == "Unknown")].index
dataSet.drop(drop_ranked_unknown, inplace=True)

# convert the column from string to integers
dataSet["Ranked"] = dataSet["Ranked"].astype(str).astype(float)

# Mamke sure that are columns are prestned right data
dataSet.describe()

# TOP 100 from the top
top15 = dataSet.nlargest(15, ['Score'])

# TOP 100 from the bottom
bot15 = dataSet.nsmallest(15, ['Score'])

# print("Unique Types of anime = ", dataSet['Type'].unique())
# print("\nUnique Genres of anime = ", dataSet["Genres"].unique())
# print("\nUnique Rating of anime = ", dataSet["Rating"].unique())

# Checking for outliers > relationship analysis

## Corralation Matrix of the ENTIRE SET,
### SCORE, RANKED, POPULARITY
# corelation = dataSet.corr()
# sns.heatmap(corelation, xticklabels=corelation.columns,yticklabels=corelation.columns, annot = True)

top15.describe()

bot15.describe()


def plots():
    sns.relplot(x="Ranked", y="Popularity", hue="Type", data=dataSet, aspect=1.5, kind="line")

    # inverses the y axis and x axis
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()

    sns.relplot(x="Popularity", y="Score", hue="Rating", data=dataSet)
    plt.gca().invert_xaxis()

    sns.relplot(x="Ranked", y="Score", hue="Rating", data=dataSet)
    plt.gca().invert_xaxis()

    # Plot comparing the ranking and score based on the top 15
    sns.relplot(x="Ranked", y="Score", hue="Name", data=top15)
    plt.gca().invert_xaxis()

    # Same plot comparing the top 15 but with a line of best fit and no legend
    sns.lmplot(x="Ranked", y="Score", ci=None, data=top15)
    plt.gca().invert_xaxis()

    # Plot comparing the ranking and score based on the botom 15
    sns.relplot(x="Ranked", y="Score", hue="Name", data=bot15)
    plt.gca().invert_xaxis()

    # Same plot comparing the bottom 15 but with a line of best fit and no legend
    sns.lmplot(x="Ranked", y="Score", ci=None, data=bot15)
    plt.gca().invert_xaxis()


def lasso():
    # Setting the training and testing data
    trainData, testData = train_test_split(dataSet, test_size=0.2)

    # Create poor linear model
    poorModel = linear_model.LinearRegression()

    # Train linear model
    x = np.c_[dataSet.iloc[:, 7]]
    y = np.c_[dataSet.iloc[:, 1]]
    poorModel.fit(x, y)

    # Plot points for graph
    plt.scatter(x, y, color='k')
    plt.xlabel("Ranked")
    plt.ylabel("Score")

    # Plotting the poor model
    model_features = PolynomialFeatures(degree=2, include_bias=False)
    # Setting the parameters for the axis
    plt.yticks(np.arange(0, 50, 3))
    plt.xticks(np.arange(0, 26, 2.5))

    # Fit graph to x axis data
    xPoly = model_features.fit_transform(x)
    # Graph as a linear regression
    linRegression = linear_model.LinearRegression()
    linRegression.fit(xPoly, y)

    # Adapt data to linear regression line
    newX = np.linspace(7, 25, 100).reshape(100, 1)
    newPolyx = model_features.transform(newX)
    newY = linRegression.predict(newPolyx)
    plt.plot(newX, newY, "m-", linewidth=2, label="Linear Regression")

    # Calculate poor MSE
    poorPredict = linRegression.predict(xPoly)
    poorMSE = mean_squared_error(y, poorPredict)

    # Create the legend and title of the graph
    plt.legend()
    plt.title("Model: Score vs. Ranked")

    # Training the better model
    ridge_reg = Ridge(alpha=1, solver="sag", random_state=42)
    ridge_reg.fit(x, y)
    lasso_reg = Lasso(alpha=0.1, random_state=42)
    lasso_reg.fit(x, y)
    lassoY = ridge_reg.predict(newX)

    # Calculate good MSE
    goodPredict = ridge_reg.predict(x)
    goodMSE = mean_squared_error(y, goodPredict)

    # Plotting the lasso line
    plt.plot(newX, lassoY, "c-", linewidth=2, label="Lasso")
    plt.legend()

    # Print the MSEs
    print(poorMSE)
    print(goodMSE)


droppedSet = top15.copy()
del droppedSet["Name"]
del droppedSet["Type"]
del droppedSet["Episodes"]
del droppedSet["Studios"]
del droppedSet["Source"]
del droppedSet["Rating"]

poorModel = linear_model.LinearRegression()

# Train linear model
x = np.c_[dataSet.iloc[:, 7]]
y = np.c_[dataSet.iloc[:, 1]]
poorModel.fit(x, y)

# Plot points for graph
plt.scatter(x, y, color='k')
plt.xlabel("Ranked")
plt.ylabel("Score")

# Plotting the poor model
model_features = PolynomialFeatures(degree=2, include_bias=False)
# Setting the parameters for the axis
plt.yticks(np.arange(0, 50, 3))
plt.xticks(np.arange(0, 26, 2.5))

# Fit graph to x axis data
xPoly = model_features.fit_transform(x)
# Graph as a linear regression
linRegression = linear_model.LinearRegression()
linRegression.fit(xPoly, y)

# Adapt data to linear regression line
newX = np.linspace(0, 25, 100).reshape(100, 1)
newPolyx = model_features.transform(newX)
newY = linRegression.predict(newPolyx)
plt.plot(newX, newY, "m-", linewidth=2, label="Linear Regression")
