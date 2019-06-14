from plugins.database import db
from datetime import datetime
from models.base.model import BaseModel
from flask_login import UserMixin


class User(BaseModel, UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    spotify_id = db.Column(db.String(255), unique=True, nullable=False)
    name = db.Column(db.String(80), unique=True, nullable=False)
    jwt = db.Column(db.String(255), unique=True, nullable=False)
    created_on = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


    def to_json(self):
        return {
                'spotify_id': self.spotify_id,
                'name': self.name,
                }

