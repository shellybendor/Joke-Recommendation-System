import numpy as np 
import pandas as pd
from pathlib import Path
import random
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
# def get_user_item_matrix():
#     df = pd.read_excel(Path("..", "FINAL jester 2006-15.xls"))
#     return df

# def add_user_to_excel(user_tag, user_id): # TODO: find way to set user_id
#     user_df = pd.read_excel(Path("..", "Users.xlsx"))
#     new_user = pd.DataFrame([[user_tag, user_id]], columns=['user_tag', 'user_id'])
#     added = pd.concat([user_df, new_user], ignore_index=True)
#     added.to_excel(Path("..", "Users.xlsx"), index=False)




class DatabaseWrapper:
    NOT_RATED = 99

    def __init__(self, user_name: str):
        self._user_item_df = pd.read_csv("user_item_matrix.csv")
        self._users_df = pd.read_csv("all_users.csv")
        self._jokes_df = pd.read_csv("all_jokes.csv")
        self._current_max_user_index = self._user_item_df.shape[0] - 1
        self._num_jokes = self._jokes_df.shape[0]
        if not self._check_if_user_exists(user_name):
            self._add_new_user(user_name)
        self._current_user_index = int(self._users_df[self._users_df.user_name == user_name]['user_index'])

    def _check_if_user_exists(self, user_name: str):
        return user_name in self._users_df["user_name"].values

    def _add_new_user(self, user_name: str):
        self._current_max_user_index += 1
        new_user = pd.DataFrame([[user_name, self._current_max_user_index]], columns=['user_name', 'user_index'])
        self._users_df = pd.concat([self._users_df, new_user], ignore_index=True)
        self._users_df.reset_index()
        new_rating_arr = [self.NOT_RATED for _ in range(self._num_jokes)]
        new_rating_arr.insert(0, 0)  # setting number of ratings for new user
        self._user_item_df.loc[len(self._user_item_df)] = new_rating_arr
        self._user_item_df.reset_index()
    
    def _get_random_new_joke(self):
        found_new_joke = False
        while not found_new_joke:
            joke_num = random.randint(0, self._num_jokes - 1)
            if self._user_item_df[f"joke_{joke_num}"][self._current_user_index] == self.NOT_RATED:
                found_new_joke = True
        return joke_num, self._jokes_df["jokes"][joke_num]
    
    def _add_joke_rating(self, joke_num: int, rating: int):
        self._user_item_df[f"joke_{joke_num}"][self._current_user_index] = rating

    def save_changes_to_db(self):
        self._user_item_df.to_csv("user_item_matrix.csv")
        self._users_df.to_csv("all_users.csv")

    def _get_recommended_joke(self):
        pass # TODO: implement matrix factorization

    def get_next_joke(self):
        pass # TODO: checks how many jokes have been rated and get next one in accordance

    def _get_num_jokes_rated(self):
        return self._user_item_df["num_ratings"][self._current_user_index]



tmp = DatabaseWrapper(user_name="shelly")
num, joke = tmp._get_random_new_joke()
print(joke)
rating = int(input("how was the joke?"))
tmp._add_joke_rating(num, rating)