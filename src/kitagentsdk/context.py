# src/kitagentsdk/context.py
import socket
import json
import logging

logger = logging.getLogger(__name__)

class ContextClient:
    """
    A client to send messages (logs, events) to the kitexec context server
    via a Unix domain socket.
    """
    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self.socket = None
        self._connect()

    def _connect(self):
        try:
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.socket.connect(self.socket_path)
        except (socket.error, FileNotFoundError) as e:
            logger.error(f"Could not connect to kitexec context socket at {self.socket_path}: {e}")
            self.socket = None

    def send_message(self, payload: dict):
        if not self.socket:
            return
        try:
            message = json.dumps(payload) + '\0' # Use null char as delimiter
            self.socket.sendall(message.encode('utf-8'))
        except socket.error as e:
            logger.error(f"Failed to send message to context socket: {e}")
            # Attempt to reconnect on next message
            self.socket.close()
            self.socket = None
            self._connect()

    def log(self, message: str):
        self.send_message({"type": "log", "payload": message})

    def emit_event(self, event_name: str, status: str = "info"):
        self.send_message({"type": "event", "event": event_name, "status": status})

    def close(self):
        if self.socket:
            self.socket.close()
            self.socket = None