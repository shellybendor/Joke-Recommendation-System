import random
import re
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matrix_factorization import KernelMF, train_update_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns


class JokeRecommender:
    NOT_RATED = 99

    def __init__(self):
        print("STARTING INIT")
        self._user_item_df = pd.read_csv("user_item_matrix.csv", dtype=np.float32)
        self._users_df = pd.read_csv("all_users.csv")
        self._jokes_df = pd.read_csv("all_jokes.csv")
        self._current_max_user_index = self._user_item_df.shape[0] - 1
        self._num_jokes = self._jokes_df.shape[0]
        self._mf = KernelMF(n_epochs=20, n_factors=40, verbose=0, lr=0.001, reg=0.005, min_rating=-10, max_rating=10)
        self._ratings = self._preprocess_data(self._user_item_df)
        self._mf.fit(self._ratings[["user_id", "item_id"]], self._ratings["rating"])
        print("DONE INIT")
    
    def _save_content_matrix(self):
        common_words = pd.read_csv("common_words_in_jokes.csv")
        ind = pd.RangeIndex(start=self._user_item_df.shape[0], stop=self._user_item_df.shape[0] + common_words.shape[0], step=1)
        content_df = pd.DataFrame(columns=self._user_item_df.columns, index=ind)
        index_of_word = self._user_item_df.shape[0]
        for word in common_words["word"]:
            for joke_num in range(self._num_jokes):
                if word in self._jokes_df["jokes"][joke_num].lower():
                    content_df.loc[index_of_word, f"joke_{joke_num + 1}"] = 10
            index_of_word += 1
        self._content_df = self._preprocess_data(content_df)
                

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
            print(self._users_df)
    
    def _get_random_new_joke(self, user_id: int):
        unheard_jokes = self._user_item_df.columns[(self._user_item_df == self.NOT_RATED).iloc[user_id]].tolist()
        new_joke = random.choice(unheard_jokes)
        joke_num = int(re.findall(r'\d+', new_joke)[0])
        return joke_num, self._jokes_df["jokes"][joke_num]
    
    def _add_joke_rating(self, user_name: str, joke_num: int, rating: int):
        user_ind = self._get_user_index(user_name)
        if self._user_item_df[f"joke_{joke_num}"][user_ind] != self.NOT_RATED:
            raise Exception(f"Joke {joke_num} has already been rated by user with id: {user_ind}")
        self._user_item_df.loc[user_ind, f"joke_{joke_num}"] = rating
        self._user_item_df.loc[user_ind, "num_ratings"] = self._user_item_df["num_ratings"][user_ind] + 1
        self._ratings.loc[len(self._ratings)] = np.array([user_ind, f"joke_{joke_num}", rating])
        self._ratings = self._ratings.astype({"rating": np.float32, "user_id": np.int64})
        ratings_for_update = self._ratings[self._ratings["user_id"].astype(int) == user_ind]
        self._mf.update_users(
            ratings_for_update[["user_id", "item_id"]], ratings_for_update["rating"], lr=0.001, n_epochs=20, verbose=0
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
        counts = self._ratings["user_id"].value_counts()
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

        if with_content:
            self._save_content_matrix()
            X_train_initial = pd.concat([X_train_initial, self._content_df[["user_id", "item_id"]]], ignore_index=True)
            y_train_initial = pd.concat([y_train_initial, self._content_df["rating"]], ignore_index=True).astype(np.float32)

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

    def _get_recommended_joke(self, user: int):
        items_known = self._ratings.query("user_id == @user")["item_id"]
        best = self._mf.recommend(user=user, items_known=items_known, amount=1, include_user=False)
        joke_num = best["item_id"].item().replace("joke_", "")
        return joke_num, self._jokes_df["jokes"][int(joke_num) - 1]

    def get_joke(self, user_name: str):
        user = self._get_user_index(user_name)
        if self._get_num_jokes_rated(user) < 10:
            print("Here is a random joke:\n")
            return self._get_random_new_joke(user)
        else:
            print("Here is a recommened joke:\n")
            return self._get_recommended_joke(user)

    def _get_num_jokes_rated(self, user: int):
        return self._user_item_df["num_ratings"][user]



# recommender = JokeRecommender()
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
#     recommender.evaluate_model()
#     recommender.evaluate_model(with_content=True)
