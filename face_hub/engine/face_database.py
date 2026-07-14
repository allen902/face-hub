"""
Face database module.
Stores and manages registered person records and face embeddings.
Built-in version tracking lets FaceRecognizer caches invalidate efficiently.

Security notes:
  - Embeddings are stored as .npy (numpy native format), NOT pickle, to
    prevent arbitrary code execution via deserialization.
  - image_path values are validated to prevent path-traversal attacks.
  - Database files are written atomically (temp-file + rename) to prevent
    corruption on crash.
"""

import json
import os
import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from face_hub.exceptions import DatabaseError

logger = logging.getLogger("face_hub.database")

# Supported encoding file extensions
_NPY_EXT = ".npy"
_PKL_EXT = ".pkl"


class FaceDatabase:
    """Face database — manages registered persons and their embeddings."""

    def __init__(self, db_path="face_db.json", encoding_path="encodings.npy"):
        self.db_path = Path(db_path)
        self.encoding_path = Path(encoding_path)
        self.persons = []          # [{"name": str, "image_path": str}, ...]
        self.encodings = []        # list of np.ndarray
        self._version = 0
        self._cached_names = []
        self._safe_dir = self.db_path.parent.resolve()
        self.load()

    @property
    def version(self):
        """Database version — used by recognizer cache invalidation."""
        return self._version

    def _validate_image_path(self, image_path: str) -> Path:
        """Validate that image_path resolves within the safe directory."""
        resolved = Path(image_path).resolve()
        if not resolved.is_relative_to(self._safe_dir):
            raise ValueError(
                f"image_path '{image_path}' is outside the database directory "
                f"'{self._safe_dir}'"
            )
        return resolved

    def add_person(self, name: str, image_path: str, encoding: np.ndarray):
        """Add a person record.

        Args:
            name: Unique person name (non-empty string).
            image_path: Path to reference photo.
            encoding: 512-dim L2-normalized face embedding.

        Raises:
            ValueError: If name is empty, encoding shape is wrong, or
                        image_path escapes the safe directory.
        """
        if not name or not isinstance(name, str):
            raise ValueError("name must be a non-empty string")
        if not isinstance(encoding, np.ndarray) or encoding.shape != (512,):
            raise ValueError(
                f"encoding must be a numpy array of shape (512,), "
                f"got {getattr(encoding, 'shape', type(encoding))}"
            )

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
                try:
                    img_path = self._validate_image_path(person["image_path"])
                    if img_path.exists():
                        img_path.unlink(missing_ok=True)
                except ValueError:
                    logger.warning(
                        "Skipping deletion of image for '%s': path outside safe directory",
                        name,
                    )
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
                    try:
                        img_path = self._validate_image_path(person["image_path"])
                        if img_path.exists():
                            img_path.unlink(missing_ok=True)
                    except ValueError:
                        logger.warning(
                            "Skipping deletion of image for '%s': path outside safe directory",
                            name,
                        )
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
        """Persist the database to disk (atomic write).

        Writes to temporary files first, then renames to avoid corruption
        if the process is killed mid-write.
        """
        try:
            data = {"persons": self.persons}
            # Atomic JSON write
            fd, tmp_path = tempfile.mkstemp(
                suffix=".json", dir=str(self.db_path.parent)
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                Path(tmp_path).replace(self.db_path)
            except Exception:
                Path(tmp_path).unlink(missing_ok=True)
                raise

            # Atomic encoding write (.npy)
            enc_path = self.encoding_path
            if len(self.encodings) > 0:
                stacked = np.array(self.encodings, dtype=np.float32)
            else:
                stacked = np.array([], dtype=np.float32).reshape(0, 512)
            fd2, tmp_enc = tempfile.mkstemp(
                suffix=_NPY_EXT, dir=str(enc_path.parent)
            )
            try:
                os.close(fd2)
                np.save(tmp_enc, stacked)
                Path(tmp_enc).replace(enc_path)
            except Exception:
                Path(tmp_enc).unlink(missing_ok=True)
                raise

            # Set restrictive permissions (owner read/write only)
            try:
                os.chmod(self.db_path, 0o600)
                os.chmod(enc_path, 0o600)
            except OSError:
                pass  # chmod may fail on Windows; not critical

        except (IOError, OSError, ValueError) as e:
            raise DatabaseError(f"Failed to save database: {e}") from e

    def load(self):
        """Load the database from disk.

        Supports both .npy (new, safe) and legacy .pkl formats.
        Legacy .pkl files are automatically migrated to .npy on load.
        """
        try:
            if self.db_path.exists():
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.persons = data.get("persons", [])

            self.encodings = self._load_encodings()
            self._cached_names = [p["name"] for p in self.persons]
            self._version += 1
        except (json.JSONDecodeError, IOError) as e:
            raise DatabaseError(f"Failed to load database: {e}") from e

    def _load_encodings(self) -> list:
        """Load encodings from .npy file, with legacy .pkl migration."""
        enc_path = self.encoding_path

        # Try the primary path first
        if enc_path.exists():
            return self._load_npy(enc_path)

        # If the primary path is .npy, also check for a legacy .pkl file
        if enc_path.suffix == _NPY_EXT:
            legacy_path = enc_path.with_suffix(_PKL_EXT)
            if legacy_path.exists():
                return self._migrate_pkl_to_npy(legacy_path, enc_path)

        return []

    def _load_npy(self, path: Path) -> list:
        """Load encodings from a .npy file."""
        stacked = np.load(str(path), allow_pickle=False)
        if stacked.ndim == 2 and stacked.shape[0] > 0:
            return [stacked[i] for i in range(stacked.shape[0])]
        return []

    def _migrate_pkl_to_npy(self, pkl_path: Path, npy_path: Path) -> list:
        """Migrate a legacy .pkl encoding file to safe .npy format."""
        import pickle
        logger.info("Migrating legacy %s to %s", pkl_path.name, npy_path.name)
        try:
            with open(pkl_path, "rb") as f:
                encodings = pickle.load(f)
            # Save as .npy
            if encodings:
                stacked = np.array(encodings, dtype=np.float32)
            else:
                stacked = np.array([], dtype=np.float32).reshape(0, 512)
            np.save(str(npy_path), stacked)
            # Remove legacy file
            pkl_path.unlink(missing_ok=True)
            logger.info("Migration complete — legacy .pkl file removed")
            return encodings
        except Exception as e:
            logger.error("Failed to migrate legacy .pkl: %s", e)
            return []

    def clear(self):
        """Clear the database and delete persisted files."""
        for person in self.persons:
            try:
                img_path = self._validate_image_path(person["image_path"])
                if img_path.exists():
                    img_path.unlink(missing_ok=True)
            except ValueError:
                logger.warning(
                    "Skipping deletion of image for '%s': path outside safe directory",
                    person["name"],
                )

        self.persons = []
        self.encodings = []
        self._cached_names = []
        self._version += 1
        if self.db_path.exists():
            self.db_path.unlink()
        if self.encoding_path.exists():
            self.encoding_path.unlink()
        # Also clean up any legacy .pkl file
        legacy = self.encoding_path.with_suffix(_PKL_EXT)
        if legacy.exists():
            legacy.unlink(missing_ok=True)
