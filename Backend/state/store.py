from typing import Dict, Any
import threading


class InMemoryStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {
            "prices": {},
            "returns": {},
            "correlation_matrix": None,
            "last_update": None
        }

    def update(self, key: str, value: Any):
        with self._lock:
            self._data[key] = value

    def get(self, key: str):
        with self._lock:
            return self._data.get(key)

    def get_all(self):
        with self._lock:
            return self._data.copy()


store = InMemoryStore()
