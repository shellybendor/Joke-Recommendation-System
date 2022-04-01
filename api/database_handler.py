import numpy as np 
import pandas as pd
from pathlib import Path
import random
from matrix_factorization import BaselineModel, KernelMF, train_update_test_split
from sklearn.metrics import mean_squared_error
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

class MF():
    NULL = 99
    
    def __init__(self, R, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[self.R != self.NULL])

        # Create a list of training samples
        import ipdb; ipdb.set_trace()
        self.samples = [(i, j, self.R[f"joke_{j}"][i]) for i in range(self.num_users) for j in range(1, self.num_items + 1) if self.R[f"joke_{j}"][i] != self.NULL]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i+1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, mse))

        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = np.where(self.R == self.NULL)
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic gradient descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)


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
        if self._user_item_df[f"joke_{joke_num}"][self._current_user_index] != self.NOT_RATED:
            raise Exception(f"Joke {joke_num} has already been rated by user with id: {self._current_user_index}")
        self._user_item_df[f"joke_{joke_num}"][self._current_user_index] = rating

    def save_changes_to_db(self):
        self._user_item_df.to_csv("user_item_matrix.csv")
        self._users_df.to_csv("all_users.csv")
    
    def matrix_factorization(self):
        # latent_features_num = 2  # TODO: calculate best number for this
        jokes_only = self._user_item_df.drop('num_ratings', axis=1)
        jokes_only.replace(self.NOT_RATED, np.nan, inplace=True)
        jokes_only = jokes_only[~np.isnan(jokes_only)]
        import ipdb; ipdb.set_trace()
        # mf = MF(jokes_only, K=latent_features_num, alpha=0.1, beta=0.01, iterations=20)
        # print(mf.train())
        # predicted = mf.full_matrix()
        (
            X_train_initial,
            y_train_initial,
            X_train_update,
            y_train_update,
            X_test_update,
            y_test_update,
        ) = train_update_test_split(jokes_only, frac_new_users=0.2)

        # Initial training
        matrix_fact = KernelMF(n_epochs=20, n_factors=100, verbose=1, lr=0.001, reg=0.005)
        matrix_fact.fit(X_train_initial, y_train_initial)

        # Update model with new users
        matrix_fact.update_users(
            X_train_update, y_train_update, lr=0.001, n_epochs=20, verbose=1
        )
        pred = matrix_fact.predict(X_test_update)
        rmse = mean_squared_error(y_test_update, pred, squared=False)
        print(f"\nTest RMSE: {rmse:.4f}")

        # Get recommendations
        user = 200
        items_known = X_train_initial.query("user_id == @user")["item_id"]
        matrix_fact.recommend(user=user, items_known=items_known)
        

    def _get_recommended_joke(self):
        pass # TODO: implement matrix factorization

    def get_next_joke(self):
        pass # TODO: checks how many jokes have been rated and get next one in accordance

    def _get_num_jokes_rated(self):
        return self._user_item_df["num_ratings"][self._current_user_index]



tmp = DatabaseWrapper(user_name="shelly")
# num, joke = tmp._get_random_new_joke()
# print(joke)
# rating = int(input("how was the joke?"))
# tmp._add_joke_rating(num, rating)
tmp.matrix_factorization()