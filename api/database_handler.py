import numpy as np 
import pandas as pd
from pathlib import Path
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn import neighbors
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler



"""An excel spreadsheet with 150 rows
The row number corresponds to the joke ID
"""
# jokes = pd.read_excel(Path("..", "Dataset3JokeSet.xlsx"))
# joke_id = 0
# print(jokes["jokes"].iloc[joke_id])

# """The data is formatted as an excel file representing a 54,905 by 151 with rows as users and columns as jokes.
# The left-most column contains the total amount of jokes rated by the user.
# """
# ratings = pd.read_excel(Path("..", "FINAL jester 2006-15.xls"))
# user_id = 0
# print(ratings[joke_id].iloc[user_id + 1])

def add_user_to_excel(user_tag, user_id): # TODO: find way to set user_id
    user_df = pd.read_excel(Path("..", "Users.xlsx"))
    new_user = pd.DataFrame([[user_tag, user_id]], columns=['user_tag', 'user_id'])
    added = pd.concat([user_df, new_user], ignore_index=True)
    added.to_excel(Path("..", "Users.xlsx"), index=False)
