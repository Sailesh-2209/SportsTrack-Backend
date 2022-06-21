from flask import Blueprint
from flask_cors import CORS

ml = Blueprint("ml", __name__)
CORS(ml)

from . import routes
