from flask import Flask, request
from flask.helpers import send_from_directory
from flask_cors import CORS, cross_origin #comment this on deployment
from joke_recommender import JokeRecommender

app = Flask(__name__, static_folder='build', static_url_path='')
CORS(app) #comment this on deployment

recommender = JokeRecommender()

@app.route('/get_joke', methods=['POST'])
@cross_origin()
def get_joke():
    print("getting joke")
    data = request.get_json()
    user = data['user']
    joke = recommender.get_joke(user)
    # print(joke)
    return {'joke': joke}


@app.route('/rate_joke', methods=['POST'])
@cross_origin()
def rate_joke():
    print("rating joke")
    data = request.get_json()
    user = data['user']
    joke_num = data['joke_num']
    rating = data['rating']
    recommender.add_joke_rating(user, int(joke_num), int(rating))
    return "Rated Joke"


@app.route('/add_user', methods=['POST'])
@cross_origin()
def add_user():
    print("setting user")
    data = request.get_json()
    user = data['user']
    recommender.add_new_user(user)
    return "Added User"

@app.route('/close_session', methods=['POST'])
@cross_origin()
def close_session():
    print("setting user")
    data = request.get_json()
    user = data['user']
    recommender.add_new_user(user)
    return "Added User"

@cross_origin()
def server():
    return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    app.run()
