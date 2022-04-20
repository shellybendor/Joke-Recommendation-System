import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from joke_recommender import JokeRecommender
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__)
CORS(app)
uri = os.environ['DATABASE_URL']
if uri and uri.startswith("postgres://"):
    uri = uri.replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_DATABASE_URI'] = uri
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)
from models import Rating

all_ratings = Rating.query.all()
rating_df = pd.DataFrame.from_dict([e.serialize() for e in all_ratings])
recommender = JokeRecommender(rating_df)

@app.route('/get_joke', methods=['POST'])
def get_joke():
    print("getting joke")
    data = request.get_json()
    ratings = Rating.query.all()
    ratings_df = pd.DataFrame.from_dict([e.serialize() for e in ratings])
    user = data['user']
    joke = recommender.get_joke(user, ratings_df)
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
    return "Rated Joke"

@app.route("/", methods=["GET"])
def see_all_added_ratings():
    ratings = Rating.query.all()
    return jsonify([e.serialize() for e in ratings])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
