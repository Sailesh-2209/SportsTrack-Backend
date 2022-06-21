from werkzeug.security import generate_password_hash, check_password_hash
from . import db

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(64), unique=True)
    username = db.Column(db.String(64), unique=True)
    password_hash = db.Column(db.String(128))
    characters = db.relationship('Session', backref='user', lazy='joined')

    @property
    def password(self):
        raise AttributeError("Raw password cannot be accessed")

    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)


class Session(db.Model):
    __tablename__ = "sessions"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True)
    duration = db.Column(db.Integer)
    vidDir = db.Column(db.String(64), unique=True)
    heatmapPath = db.Column(db.String(64), unique=True)
    pieChartPath = db.Column(db.String(64), unique=True)
    heatmapURL = db.Column(db.String(64), unique=True)
    pieChartURL = db.Column(db.String(64), unique=True)
    user = db.relationship('User')
