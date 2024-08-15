# stdlib
import logging
from logging.config import dictConfig

import socketio as sio

# third-party
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# local

dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "wsgi": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",  # <-- Solution
                "formatter": "default",
            }
        },
        "root": {"level": "INFO", "handlers": ["wsgi"]},
    }
)
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "examplesecretkey"
socketio = SocketIO(app)  # Create a Socket.IO server

app.logger.setLevel(logging.INFO)


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def handle_connect() -> None:
    app.logger.info("Client connected to server")


if __name__ == "__main__":
    other_server_socket = (
        sio.Client()
    )  # Create a Socket.IO client to connect to another server
    other_server_socket.connect("http://0.0.0.0:1024")  # Connect to the other server
    breakpoint()
    print("connet")
