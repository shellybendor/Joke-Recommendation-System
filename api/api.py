from flask import Flask, request
from flask_cors import CORS #comment this on deployment
from database_handler import add_user_to_excel


app = Flask(__name__)
CORS(app) #comment this on deployment

@app.route('/api/joke')
def get_joke():
    return {'joke': "knock knock"}

@app.route('/add_user', methods=['POST'])
def add_user():
    # print("adding user")
    data = request.get_json()
    # print(data)
    user = data['user']
    add_user_to_excel(user, 0)
    return "Success"
    


