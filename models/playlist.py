from plugins.database import db
from models.base.model import BaseModel
from datetime import datetime


class Playlist(BaseModel, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    spotify_id = db.Column(db.String(255), unique=True, nullable=False)
    name = db.Column(db.String(255), unique=True, nullable=False)
    image = db.Column(db.String(255), unique=True, nullable=False)
    created_on = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def to_json(self):
        return {
                'spotify_id': self.spotify_id,
                'name': self.name,
                'image': self.image
                }
