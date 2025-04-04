import eventlet
eventlet.monkey_patch()

from app import app, socketio

application = socketio.WSGIApp(socketio, app)  # Critical change

if __name__ == "__main__":
    socketio.run(app)
