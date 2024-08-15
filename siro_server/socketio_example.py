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
# other_server_socket = sio.Client()   # Create a Socket.IO client to connect to another server
# other_server_socket.connect('http://localhost:5000') # Connect to the other server


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def handle_connect() -> None:
    app.logger.info("Client connected to server")


def error(message):
    app.logger.error(message)
    emit("error", {"error": message})


@socketio.on("my_event")
def handle_my_event(data):
    my_string = data.get(
        "my_string", ""
    )  # Assuming the data is a dictionary, it can be any data type
    my_int = int(data.get("my_int", -1))
    my_bool = bool(data.get("my_bool", False))

    app.logger.info(f"Received data: {my_string}, {my_int}, {my_bool}")


def send_message_to_clients(message):
    emit("message", {"message": message})


# def send_message_to_other_server(message):
#     other_server_socket.emit('message', {'message': message})
#
#
# @other_server_socket.on('their_event')
# def handle_message(data):
#     message = data.get('message', "")
#     app.logger.info(f"Received message from other server: {message}")
#     send_message_to_clients(message)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Start the server")
    parser.add_argument("--host", default="0.0.0.0", help="The host to bind to")
    parser.add_argument("--port", default=1024, type=int, help="The port to bind to")
    args = parser.parse_args()

    if args.port < 0 or args.port > 65535:
        print("Invalid port number")
        return

    app.logger.info(f"Starting server on {args.host}:{args.port}")

    socketio.run(app, host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
