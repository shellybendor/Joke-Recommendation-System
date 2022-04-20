from app import db

class Rating(db.Model):
    __tablename__ = 'ratings'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(80))
    item_id = db.Column(db.String(80))
    rating = db.Column(db.Float)


    def __init__(self, user_id, item_id, rating):
        self.user_id = user_id
        self.item_id = item_id
        self.rating = rating

    def __repr__(self):
        return '<id {} heyyyy>'.format(self.id)
    
    def serialize(self):
        return {
            # 'id': self.id, 
            'user_id': self.user_id,
            'item_id': self.item_id,
            'rating':self.rating
        }