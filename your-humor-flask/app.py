import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from joke_recommender import JokeRecommender
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['DATABASE_URL']
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)
from models import Rating

recommender = JokeRecommender()

@app.route('/get_joke', methods=['POST'])
def get_joke():
    print("getting joke")
    data = request.get_json()
    ratings = Rating.query.all()
    datframe = pd.DataFrame.from_dict([e.serialize() for e in ratings])
    print(datframe)
    user = data['user']
    joke = recommender.get_joke(user, datframe)
    return {'joke': joke}


@app.route('/rate_joke', methods=['POST'])
def rate_joke():
    print("rating joke")
    data = request.get_json()
    user = data['user']
    joke_id = data['joke_id']
    rating = data['rating']
    new_rating = Rating(user, joke_id, rating)
    db.session.add(new_rating)
    db.session.commit()
    recommender.add_joke_rating(user, joke_id, int(rating))
    return "Rated Joke"


@app.route('/add_user', methods=['POST'])
def add_user():
    print("setting user")
    data = request.get_json()
    # ratings = Rating.query.filter(Rating.user_id == data['user']).all()
    # datframe = pd.DataFrame.from_dict([e.serialize() for e in ratings])
    # print(datframe)
    user = data['user']
    recommender.add_new_user(user)
    # return jsonify([e.serialize() for e in ratings])
    return "Added User"

@app.route("/", methods=["GET"])
def tmp():
    ratings = Rating.query.all()
    return jsonify([e.serialize() for e in ratings])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
