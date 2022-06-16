from . import auth
from ..models import User
from .. import db
import json
from flask import request
import time
import re
import jwt
import os
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv


load_dotenv()


def verify_password(password):
    spec_char = ["!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "_", "-", "+", "="]
    if len(password) < 6:
        return False, "Password too short"
    if len(password) > 20:
        return False, "Password too long"
    if not any(char.isdigit() for char in password):
        return False, "Password must have atleast 1 digit"
    if not any(char.isupper() for char in password):
        return False, "Password must have atleast 1 capital letter"
    if not any(char.islower() for char in password):
        return False, "Password must have atleast 1 small letter"
    if not any(char in spec_char for char in password):
        return False, "Password must have atleast 1 special character"
    return True, "OK"


def verify_email(email):
    user = User.query.filter_by(email=email).first()
    if (user != None):
        return False, "Account with this email already exists"
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if re.fullmatch(regex, email):
        return True, "OK"
    else:
        return False, "Not a valid email"


def verify_username(username):
    user = User.query.filter_by(username=username).first()
    if (user != None):
        return False, "Username is taken"
    if (len(username) < 3):
        return False, "Username too short"
    return True, "OK"


@auth.route("/signup", methods=["POST"])
def signup():
    data = json.loads(request.data)
    if not "email" in data:
        return "Email not provided", 400
    elif not "password" in data:
        return "Password not provided", 400
    elif not "username" in data:
        return "Username not provided", 400
    else:
        username = data["username"]
        password = data["password"]
        email = data["email"]
        username_validity = verify_username(username)
        password_validity = verify_password(password)
        email_validity = verify_email(email)
        if not username_validity[0]:
            return username_validity[1], 400
        elif not password_validity[0]:
            return password_validity[1], 400
        elif not email_validity[0]:
            return email_validity[1], 400
        else:
            user = User(email=email, username=username)
            user.password = password
            db.session.add(user)
            db.session.commit()
            return "Sign up successful. Login with your credentials", 200


@auth.route("/signin", methods=["POST"])
def signin():
    data = json.loads(request.data)
    err_res = {
        'error_code': None,
        'message': None
    }
    if not "email" in data:
        err_res['error_code'] = 0
        err_res['message'] = "Email not provided"
        return err_res, 400
    elif not "password" in data:
        err_res['error_code'] = 1
        err_res['message'] = "Password not provided"
    else:
        email = data["email"]
        password = data["password"]
        userinfo = User.query.filter_by(email=email).first()

        if (userinfo == None):
            err_res['error_code'] = 0
            err_res['message'] = "User does not exist"
            return err_res, 404

        ok = userinfo.verify_password(password)

        if (ok):
            exp_period = 60 * 60 * 24
            exp = int(time.time()) + exp_period
            payload = {
                "exp": exp,
                "email": email,
                "hash": generate_password_hash(password)
            }
            token = jwt.encode(payload, os.environ.get("SECRET_KEY"))
            return token, 200
        else:
            err_res['error_code'] = 1
            err_res['message'] = "Wrong password"
            return err_res, 404


@auth.route("/verifytoken", methods=["POST"])
def verifytoken():
    data = json.loads(request.data)
    if not "token" in data:
        return "Token is not provided as part of request body", 400
    else:
        token = data["token"]
        try:
            jwt.decode(token, os.environ.get("SECRET_KEY"), algorithms=["HS256"])
        except:
            return "Invalid token", 400
    return "OK", 200
