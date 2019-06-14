from flask import Flask
from web.routes.views import web
from plugins.database import db
from plugins.login import login_manager
from plugins.socket import socket_manager
import os.path


def create_app():
    app = Flask(__name__, template_folder='web/templates', static_folder='web/static', static_url_path='/static')
    app.config['DEBUG'] = True
    app.config['SECRET_KEY'] = '9OLWxND4o83j4K4iuopO'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///spotilearn.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    app.register_blueprint(web)
    login_manager.init_app(app)
    socket_manager.init_app(app)
    return app


def setup_database(app):
    with app.app_context():
        db.create_all()


if __name__ == '__main__':
    app = create_app()
    if not os.path.isfile('spotilearn.db'):
        setup_database(app)
    app.run(port=8080)
