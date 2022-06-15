from . import main

@main.route("/")
def welcome():
    return "<h1>Hello, World!</h1>"
