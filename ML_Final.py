"""
Shriji Shah (100665031)
Zachary Silver (100752283)
INFR 3700 Final Assignment
"""

# Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

# List for column headers

# Creating the DataFrame
dataSet = pd.read_csv(
    'https://raw.githubusercontent.com/OzymandiasThe2/machine_learning_final_project/main/anime.csv')
dataSet = dataSet.drop(columns=["MAL_ID", "English name", "Japanese name", "Aired", "Premiered",
                                "Producers", "Licensors", "Duration", "Members", "Favorites", "Watching",
                                "Completed", "On-Hold", "Dropped", "Plan to Watch",
                                "Score-10", "Score-9", "Score-8", "Score-7", "Score-6", "Score-5",
                                "Score-4", "Score-3", "Score-2", "Score-1"])

index_names = dataSet[(dataSet["Score"] == "Unknown")].index
dataSet.drop(index_names, inplace=True)

dataSet["Score"] = dataSet["Score"].astype(str).astype(float)

dataSet.to_csv('test.csv')

# TOP 99
top_99 = dataSet[(dataSet["Score"] < 8.52)].index

dataSet_T100 = dataSet.drop(top_99, inplace=True)

# dataSet_T100.to_csv('test_top100.csv')


# BOT 99
bot_99 = dataSet[(dataSet["Score"] < 4.31)].index
# dataSet_B100 = dataSet.drop(bot_99, inplace=True)

bot_99.to_csv('test_bot100.csv')


# TOP 100 Anime vs Bottom Anime

# bar_chart for just genres or studios

# scatter of score vs genre

# scatter of score vs studio

# index_names = dataSet[(dataSet["Score"] )].index
# dataSet.drop(index_names, inplace=True)


studios = set()
print(len(set([x.lower() for x in dataSet['Studios']])))


# plt.scatter(xList, labels, color = 'b')
# plt.xlabel("years of education")
# plt.ylabel("salary (in K$)")
# plt.show()




# Figure out how to delete rows with unknown score values

print(dataSet)

# xList = []
# labels = []
# names = []
# firstLine = True
# for line in dataSet:
#     if firstLine:
#         names = line.split(",")
#         firstLine = False
#     else:
#         #split on comma
#         row = line.split(",")
#         #put labels in separate array
#         labels.append(float(row[2]))
#         #convert row to floats
#         xList.append(float(row[3]))
#
# dataSet.close()
# # Plot points
# plt.scatter(xList, labels, color = 'b')
# plt.xlabel("years of education")
# plt.ylabel("salary (in K$)")
# plt.show()


# # x = meta_score_column
# x =
#
# # y = platforms_column
# y =
#
# plt.scatter(x, y)
# plt.show()

# print(df)
