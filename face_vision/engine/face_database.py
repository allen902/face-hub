"""
Face database module.
Stores and manages registered person records and face embeddings.
Built-in version tracking lets FaceRecognizer caches invalidate efficiently.
"""

import json
import os
import pickle
import logging
from pathlib import Path

import numpy as np

from face_vision.exceptions import DatabaseError

logger = logging.getLogger("face_vision.database")


class FaceDatabase:
    """Face database — manages registered persons and their embeddings."""

    def __init__(self, db_path="face_db.json", encoding_path="encodings.pkl"):
        self.db_path = Path(db_path)
        self.encoding_path = Path(encoding_path)
        self.persons = []          # [{"name": str, "image_path": str}, ...]
        self.encodings = []        # list of np.ndarray
        self._version = 0
        self._cached_names = []
        self.load()

    @property
    def version(self):
        """Database version — used by recognizer cache invalidation."""
        return self._version

    def add_person(self, name: str, image_path: str, encoding: np.ndarray):
        """Add a person record."""
        for person in self.persons:
            if person["name"] == name:
                return False, f"Person '{name}' already exists"

        self.persons.append({"name": name, "image_path": image_path})
        self.encodings.append(encoding)
        self._cached_names.append(name)
        self._version += 1
        self.save()
        return True, f"Added: {name}"

    def remove_person(self, name: str):
        """Remove a single person."""
        for i, person in enumerate(self.persons):
            if person["name"] == name:
                img_path = Path(person["image_path"])
                if img_path.exists():
                    img_path.unlink(missing_ok=True)
                del self.persons[i]
                del self.encodings[i]
                del self._cached_names[i]
                self._version += 1
                self.save()
                return True, f"Removed: {name}"
        return False, f"Not found: {name}"

    def remove_persons(self, names: list):
        """Remove multiple persons at once."""
        removed = []
        not_found = []
        to_remove_indices = []

        for name in names:
            found = False
            for i, person in enumerate(self.persons):
                if person["name"] == name:
                    img_path = Path(person["image_path"])
                    if img_path.exists():
                        img_path.unlink(missing_ok=True)
                    to_remove_indices.append(i)
                    removed.append(name)
                    found = True
                    break
            if not found:
                not_found.append(name)

        # Delete from the back to keep indices valid
        for i in sorted(to_remove_indices, reverse=True):
            del self.persons[i]
            del self.encodings[i]
            del self._cached_names[i]

        if removed:
            self._version += 1
        self.save()
        return removed, not_found

    def get_names(self) -> list:
        """Return all registered names."""
        return self._cached_names.copy()

    def get_encodings_and_names(self) -> tuple:
        """Return (encodings, names). Callers must not mutate the returned lists."""
        return self.encodings, self._cached_names

    def save(self):
        """Persist the database to disk."""
        try:
            data = {"persons": self.persons}
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            with open(self.encoding_path, "wb") as f:
                pickle.dump(self.encodings, f)
        except (IOError, OSError, pickle.PickleError) as e:
            raise DatabaseError(f"Failed to save database: {e}") from e

    def load(self):
        """Load the database from disk."""
        try:
            if self.db_path.exists():
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.persons = data.get("persons", [])
            if self.encoding_path.exists():
                with open(self.encoding_path, "rb") as f:
                    self.encodings = pickle.load(f)
            self._cached_names = [p["name"] for p in self.persons]
            self._version += 1
        except (json.JSONDecodeError, IOError, pickle.UnpicklingError) as e:
            raise DatabaseError(f"Failed to load database: {e}") from e

    def clear(self):
        """Clear the database and delete persisted files."""
        for person in self.persons:
            img_path = Path(person["image_path"])
            if img_path.exists():
                img_path.unlink(missing_ok=True)

        self.persons = []
        self.encodings = []
        self._cached_names = []
        self._version += 1
        if self.db_path.exists():
            self.db_path.unlink()
        if self.encoding_path.exists():
            self.encoding_path.unlink()
