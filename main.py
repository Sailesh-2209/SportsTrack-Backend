import os
from app import create_app, db
from app.models import *
from flask_migrate import Migrate
from dotenv import load_dotenv

load_dotenv()

CONFIG = os.environ.get("FLASK_CONFIG") or "default"

app = create_app(CONFIG)
migrate = Migrate(app, db)

@app.shell_context_processor
def make_shell_context():
    return dict(db=db)

if __name__ == "__main__":
    app.run()
