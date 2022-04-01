import numpy as np 
import pandas as pd
from pathlib import Path
import random
from matrix_factorization import BaselineModel, KernelMF, train_update_test_split
from sklearn.metrics import mean_squared_error


class DatabaseWrapper:
    NOT_RATED = 99

    def __init__(self):
        self._user_item_df = pd.read_csv("user_item_matrix.csv")
        self._users_df = pd.read_csv("all_users.csv")
        self._jokes_df = pd.read_csv("all_jokes.csv")
        self._current_max_user_index = self._user_item_df.shape[0] - 1
        self._num_jokes = self._jokes_df.shape[0]
        self._mf = KernelMF(n_epochs=20, n_factors=20, verbose=0, lr=0.001, reg=0.005, min_rating=-10, max_rating=10)
        self._ratings = self._preprocess_data()
        self._mf.fit(self._ratings[["user_id", "item_id"]], self._ratings["rating"])
    
    def _get_user_index(self, user_name: str) -> int:
        if not self._check_if_user_exists(user_name):
            self._add_new_user(user_name)
        return int(self._users_df[self._users_df.user_name == user_name]['user_index'])

    def _check_if_user_exists(self, user_name: str):
        return user_name in self._users_df["user_name"].values

    def add_new_user(self, user_name: str):
        if not self._check_if_user_exists(user_name):
            self._current_max_user_index += 1
            new_user = pd.DataFrame([[user_name, self._current_max_user_index]], columns=['user_name', 'user_index'])
            self._users_df = pd.concat([self._users_df, new_user], ignore_index=True)
            self._users_df.reset_index()
            new_rating_arr = [self.NOT_RATED for _ in range(self._num_jokes)]
            new_rating_arr.insert(0, 0)  # setting number of ratings for new user
            self._user_item_df.loc[len(self._user_item_df)] = new_rating_arr
            self._user_item_df.reset_index()
    
    def _get_random_new_joke(self, user_name: str):
        # TODO: make this better, randomize one choice of not null
        found_new_joke = False
        while not found_new_joke:
            joke_num = random.randint(1, self._num_jokes)
            if self._user_item_df[f"joke_{joke_num}"][self._get_user_index(user_name)] == self.NOT_RATED:
                found_new_joke = True
        return joke_num, self._jokes_df["jokes"][joke_num]
    
    def _add_joke_rating(self, user_name: str, joke_num: int, rating: int):
        user_id = self._get_user_index(user_name)
        if self._user_item_df[f"joke_{joke_num}"][user_id] != self.NOT_RATED:
            raise Exception(f"Joke {joke_num} has already been rated by user with id: {user_id}")
        self._user_item_df.loc[f"joke_{joke_num}", user_id] = rating
        self._user_item_df.loc["num_rating", user_id] += 1
        self._rating.loc[len(self._rating)] = np.array([user_id, joke_num, rating])
        ratings_for_update = self._ratings.query("user_id == @user")
        self._mf.update_users(
            ratings_for_update[["user_id", "item_id"]], ratings_for_update["rating"], lr=0.001, n_epochs=20, verbose=0
        )

    def save_changes_to_db(self):
        self._user_item_df.to_csv("user_item_matrix.csv")
        self._users_df.to_csv("all_users.csv")

    def _preprocess_data(self):
        user_items_ratings_df = self._user_item_df.drop('num_ratings', axis=1)
        user_items_ratings_df.replace(self.NOT_RATED, np.nan, inplace=True)
        user_items_ratings_df = user_items_ratings_df.stack().reset_index().rename(columns={'level_0': 'user_id', 'level_1': 'item_id', 0: 'rating'})
        return user_items_ratings_df

    def evaluate_model(self):
        data_df = self._preprocess_data()
        # Splitting data into existing user ratings for training, new user's ratings for training, and new user's ratings for testing.
        (
            X_train_initial,
            y_train_initial,
            X_train_update,
            y_train_update,
            X_test_update,
            y_test_update,
        ) = train_update_test_split(data_df, frac_new_users=0.2)

        # Initial training
        matrix_fact = KernelMF(n_epochs=20, n_factors=20, verbose=1, lr=0.001, reg=0.005, min_rating=-10, max_rating=10)
        matrix_fact.fit(X_train_initial, y_train_initial)

        # Update model with new users
        matrix_fact.update_users(
            X_train_update, y_train_update, lr=0.001, n_epochs=20, verbose=1
        )

        # Test model on predictions for "new" users
        pred = matrix_fact.predict(X_test_update)
        rmse = mean_squared_error(y_test_update, pred, squared=False)
        print(f"\nTest RMSE: {rmse:.4f}")        

    def _get_recommended_joke(self, user: int):
        items_known = self._ratings.query("user_id == @user")["item_id"]
        best = self._mf.recommend(user=user, items_known=items_known, amount=1, include_user=False)
        import ipdb; ipdb.set_trace()
        joke_num = best["item_id"].item().replace("joke_", "")
        return joke_num, self._jokes_df["jokes"][int(joke_num)]
        # TODO: extract from here the most recommended joke and its id

    def get_joke(self, user_name: str):
        user = self._get_user_index(user_name)
        if self._get_num_jokes_rated(user) < 10:
            return self._get_random_new_joke(user)
        else: 
            return self._get_recommended_joke(user)

    def _get_num_jokes_rated(self, user: int):
        return self._user_item_df["num_ratings"][user]



tmp = DatabaseWrapper()
tmp.add_new_user("shelly")
while True:
    num, joke = tmp.get_joke("shelly")
    print(joke)
    rating = int(input("how was the joke?"))
    tmp._add_joke_rating("shelly", num, rating)