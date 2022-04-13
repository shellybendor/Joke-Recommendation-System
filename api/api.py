from flask import Flask, request
from flask_cors import CORS #comment this on deployment
from joke_recommender import JokeRecommender


app = Flask(__name__)
CORS(app) #comment this on deployment

recommender = JokeRecommender()

@app.route('/api/get_joke', methods=['POST'])
def get_joke():
    print("getting joke")
    data = request.get_json()
    user = data['user']
    joke = recommender.get_joke(user)
    print(joke)
    return {'joke': joke}


@app.route('/api/rate_joke', methods=['POST'])
def rate_joke():
    print("getting joke")
    data = request.get_json()
    user = data['user']
    joke_num = data['joke_num']
    rating = data['rating']
    print(user, joke_num, rating)
    return "Success"


@app.route('/api/add_user', methods=['POST'])
def add_user():
    print("setting user")
    data = request.get_json()
    user = data['user']
    recommender.add_new_user(user)
    return "Success"

# TODO: add route for logout + saving changes to db

if __name__ == '__main__':
    app.run()
    


