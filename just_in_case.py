# -*- coding: utf-8 -*-
"""Final Project ML

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Sr2VIOGRCJzmlmsQqE25BQoJrt2i3C-w
"""

# Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import seaborn as sns

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

print("Unique Types of anime = ",data['Type'].unique())
print("\nUnique Genres of anime = ",data["Genres"].unique())
print("\nUnique Rating of anime = ",data["Rating"].unique())

# cleaning up the data
## Confirm if any null columns
data.isnull().sum()

# dropping values

dataSet = data.drop(columns=["MAL_ID", "English name", "Japanese name", "Aired", "Premiered",
                                "Producers", "Licensors", "Duration", "Members", "Favorites", "Watching",
                                "Completed", "On-Hold", "Dropped", "Plan to Watch",
                                "Score-10", "Score-9", "Score-8", "Score-7", "Score-6", "Score-5",
                                "Score-4", "Score-3", "Score-2", "Score-1"])


dataSet.head()

# shows only Popularity as ints, therefore we have to turn the rest of the int columns into actula intgers
dataSet.describe()

# drop the "unknown" values from the column
drop_score_unknown = dataSet[(dataSet["Score"] == "Unknown")].index
dataSet.drop(drop_score_unknown, inplace=True)

# convert the column from string to integers
dataSet["Score"] = dataSet["Score"].astype(str).astype(float)

# drop the "unknown" values from the column
drop_ranked_unknown = dataSet[(dataSet["Ranked"] == "Unknown")].index
dataSet.drop(drop_ranked_unknown, inplace=True)

# convert the column from string to integers
dataSet["Ranked"] = dataSet["Ranked"].astype(str).astype(float)

# Mamke sure that are columns are prestned right data
dataSet.describe()

# Since the data here is vastly large with 50k entires, we'll compare data points of the top 100 titles and the bottom 100 titles


# TOP 100 from the top
top100 = dataSet.nlargest(100, ['Score'])

# TOP 100 from the bottom
bot100 = dataSet.nsmallest(100, ['Score'])

# Checking for outliers > relationship analysis

## Corralation Matrix of the ENTIRE SET,
### SCORE, RANKED, POPULARITY
corelation = dataSet.corr()
sns.heatmap(corelation, xticklabels=corelation.columns,yticklabels=corelation.columns
            , annot = True)

## Corralation Matrix of the TOP 100,
### SCORE, RANKED, POPULARITY
corelation = top100.corr()
sns.heatmap(corelation, xticklabels=corelation.columns,yticklabels=corelation.columns
            , annot = True)

## Corralation Matrix of the BOT 100,
### SCORE, RANKED, POPULARITY
corelation = bot100.corr()

sns.heatmap(corelation, xticklabels=corelation.columns,yticklabels=corelation.columns
            , annot = True)

sns.pairplot(dataSet)

sns.pairplot(top100)

sns.pairplot(bot100)

top100.describe()

bot100.describe()

sns.relplot(x = "Popularity", y="Score", hue="Rating", data=dataSet)

sns.relplot(x = "Score", y="Popularity", hue="Rating", data=top100)

sns.relplot(x = "Score", y="Popularity", hue="Rating", data=bot100)

sns.displot(top100["Score"])

sns.displot(bot100["Score"])

sns.displot(top100["Popularity"],bins = 5)

sns.displot(bot100["Popularity"], bins = 5)

# sns.catplot(x="Type", kind= "")
