from . import main
from ..models import User

@main.route("/", methods=["GET"])
def welcome():
    return "<h1>Hello, World!</h1>"

@main.route("/users", methods=["GET"])
def get_users():
    pass
