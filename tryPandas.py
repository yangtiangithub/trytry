import pandas as pd

california_housing_dataframe = pd.read_csv("california_housing_train.csv", sep=",")
print(california_housing_dataframe.describe())
print(california_housing_dataframe.head())
print(california_housing_dataframe["longitude"][0:3])
print(california_housing_dataframe[0:3])
print(california_housing_dataframe.columns)