import random
from re import X
from tkinter import N
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matrix_factorization import KernelMF, train_update_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')


class JokeRecommender:
    NOT_RATED = 99

    def __init__(self):
        self._user_item_df = pd.read_csv("user_item_matrix.csv", dtype=np.float32)
        self._users_df = pd.read_csv("all_users.csv")
        self._jokes_df = pd.read_csv("all_jokes.csv")
        self._current_max_user_index = self._user_item_df.shape[0] - 1
        self._num_jokes = self._jokes_df.shape[0]
        self._mf = KernelMF(n_epochs=20, n_factors=40, verbose=0, lr=0.001, reg=0.005, min_rating=-10, max_rating=10)
        self._ratings = self._preprocess_data(self._user_item_df)
        self._mf.fit(self._ratings[["user_id", "item_id"]], self._ratings["rating"])
    
    def _save_content_matrix(self):
        # stop_words = set(stopwords.words('english'))
        # joke_words = ' '.join(self._jokes_df['jokes']).lower().split()
        # filtered_words = [w for w in joke_words if not w in stop_words]
        # filtered_words = pd.Series(filtered_words).value_counts()[350:500]
        # print(filtered_words.to_string())
        self._content_df = self._user_item_df.copy()
        common_words = pd.read_csv("common_words_in_jokes.csv")
        for word in common_words["word"]:
            content_ratings_arr = [self.NOT_RATED for _ in range(self._num_jokes + 1) ]
            for joke_num in range(self._jokes_df["jokes"].shape[0]):
                if word in self._jokes_df["jokes"][joke_num].lower():
                    content_ratings_arr[joke_num + 1] = 10
            self._content_df.loc[len(self._content_df)] = content_ratings_arr
                

    def _get_user_index(self, user_name: str) -> int:
        if not self._check_if_user_exists(user_name):
            self.add_new_user(user_name)
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
    
    def _get_random_new_joke(self, user_id: int):
        # TODO: make this better, randomize one choice of not null
        found_new_joke = False
        while not found_new_joke:
            joke_num = random.randint(1, self._num_jokes)
            if self._user_item_df[f"joke_{joke_num}"][user_id] == self.NOT_RATED:
                found_new_joke = True
        return joke_num, self._jokes_df["jokes"][joke_num]
    
    def _add_joke_rating(self, user_name: str, joke_num: int, rating: int):
        user_ind = self._get_user_index(user_name)
        if self._user_item_df[f"joke_{joke_num}"][user_ind] != self.NOT_RATED:
            raise Exception(f"Joke {joke_num} has already been rated by user with id: {user_ind}")
        self._user_item_df.loc[user_ind, f"joke_{joke_num}"] = rating
        self._user_item_df.loc[user_ind, "num_ratings"] = self._user_item_df["num_ratings"][user_ind] + 1
        self._ratings.loc[len(self._ratings)] = np.array([user_ind, f"joke_{joke_num}", rating])
        ratings_for_update = self._ratings[self._ratings["user_id"].astype(int) == user_ind]
        self._mf.update_users(
            ratings_for_update[["user_id", "item_id"]], ratings_for_update["rating"].astype(np.float32), lr=0.001, n_epochs=20, verbose=0
        )

    def save_changes_to_db(self):
        self._user_item_df.to_csv("user_item_matrix.csv", index=False)
        self._users_df.to_csv("all_users.csv", index=False)

    def _preprocess_data(self, df):
        new_df = df.drop('num_ratings', axis=1)
        new_df.replace(self.NOT_RATED, np.nan, inplace=True)
        new_df = new_df.stack().reset_index().rename(columns={'level_0': 'user_id', 'level_1': 'item_id', 0: 'rating'})
        return new_df

    def evaluate_model(self, with_content=False):
        eval_df = self._ratings
        if with_content:
            self._save_content_matrix()
            eval_df = self._preprocess_data(self._content_df)
        counts = eval_df["user_id"].value_counts()
        eval_data = self._ratings[self._ratings.user_id.isin(counts.index[counts.gt(1)])]
        # Splitting data into existing user ratings for training, new user's ratings for training, and new user's ratings for testing.
        (
            X_train_initial,
            y_train_initial,
            X_train_update,
            y_train_update,
            X_test_update,
            y_test_update,
        ) = train_update_test_split(eval_data, frac_new_users=0.2)

        # Initial training
        matrix_fact = KernelMF(n_epochs=20, n_factors=40, verbose=1, lr=0.001, reg=0.005, min_rating=-10, max_rating=10)
        matrix_fact.fit(X_train_initial, y_train_initial)

        # Update model with new users
        matrix_fact.update_users(
            X_train_update, y_train_update, lr=0.001, n_epochs=20, verbose=1
        )

        # Test model on predictions for "new" users
        pred = matrix_fact.predict(X_test_update)

        # Create scatter plot of predictions
        x = y_test_update.head(200)
        y = pred[:200]
        sns.regplot(x=x, y=y)
        plt.plot([-10, 10], [-10, 10], '-')
        plt.title('Scatter plot of 200 predictions')
        plt.xlabel('Test set ratings')
        plt.ylabel('Test set predicted ratings')
        plt.show()

        rmse = mean_squared_error(y_test_update, pred, squared=False)
        norm_rmse = rmse / 20
        print(f"\nTest RMSE: {rmse:.4f}")
        print(f"\nTest Normalized RMSE: {norm_rmse:.4f}")

    def _get_recommended_joke(self, user: str):
        items_known = self._ratings.query("user_id == @user")["item_id"]
        best = self._mf.recommend(user=user, items_known=items_known, amount=1, include_user=False)
        joke_num = best["item_id"].item().replace("joke_", "")
        return joke_num, self._jokes_df["jokes"][int(joke_num)]

    def get_joke(self, user_name: str):
        user = self._get_user_index(user_name)
        if self._get_num_jokes_rated(user) < 10:
            print("Here is a random joke:\n")
            return self._get_random_new_joke(user)
        else:
            print("Here is a recommened joke:\n")
            return self._get_recommended_joke(user_name)

    def _get_num_jokes_rated(self, user: int):
        return self._user_item_df["num_ratings"][user]



recommender = JokeRecommender()
# username = input("Welcome to the joke recommender! What is your name? \n")
# recommender.add_new_user(username)
# print(f"Welcome {username}!")
# keep_going = True
# while keep_going:
#     num, joke = recommender.get_joke(username)
#     print(joke, end="\n\n")
#     rating = float(input("How would you rate the joke from -10.0 to 10? "))
#     recommender._add_joke_rating(username, num, rating)
#     answer = input("Want another joke? y/n ")
#     keep_going = (answer == "y")
# print("Thanks for taking part. Please wait while we save everything :)")
# recommender.save_changes_to_db()
# answer = input("Evaluate Model? y/n ")
# if (answer == "y"):
recommender.evaluate_model()
recommender.evaluate_model(with_content=True)
