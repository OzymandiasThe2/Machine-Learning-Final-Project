"""
Shriji Shah (100665031)
Zachary Silver (100752283)
INFR 3700 Final Assignment
"""

#Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

#List for column headers

#Creating the DataFrame
dataSet = pd.read_csv('https://raw.githubusercontent.com/OzymandiasThe2/machine_learning_final_project/main/metacritic_18.07.2021_csv.csv')
#dataSet = dataSet.drop(columns = [""]



print(dataSet)