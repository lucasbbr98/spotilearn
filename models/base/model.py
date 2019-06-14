from plugins.database import db


class BaseModel:
    def save(self):
        try:
            db.session.add(self)
            db.session.commit()
            return True
        except:
            return False
