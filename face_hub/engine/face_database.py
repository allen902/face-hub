"""
Face database module.
Stores and manages registered person records and face embeddings.
Built-in version tracking lets FaceRecognizer caches invalidate efficiently.

Security notes:
  - Embeddings are stored as .npy (numpy native format), NOT pickle, to
    prevent arbitrary code execution via deserialization. Legacy .pkl
    files are only migrated when allow_legacy_pickle=True.
  - image_path values are validated to prevent path-traversal attacks.
  - Removing a person never deletes their source photo unless the caller
    explicitly passes delete_image=True.
  - Database files are written atomically (temp-file + rename) to prevent
    corruption on crash.
"""

import json
import os
import logging
import tempfile
import threading
from pathlib import Path

import numpy as np

from face_hub.exceptions import DatabaseError, SerializationError

logger = logging.getLogger("face_hub.database")

# Supported encoding file extensions
_NPY_EXT = ".npy"
_PKL_EXT = ".pkl"


class FaceDatabase:
    """Face database — manages registered persons and their embeddings."""

    def __init__(self, db_path="face_db.json", encoding_path="encodings.npy",
                 allow_legacy_pickle=False):
        """
        Args:
            db_path: Path to the JSON person registry.
            encoding_path: Path to the .npy embedding store. A legacy ".pkl"
                suffix is transparently rewritten to ".npy".
            allow_legacy_pickle: If True, a legacy pickle encoding file found
                next to encoding_path is migrated to .npy on load. Defaults to
                False because pickle deserialization can execute arbitrary
                code — only enable for files you fully trust.
        """
        self.db_path = Path(db_path)
        self.encoding_path = Path(encoding_path)
        if self.encoding_path.suffix.lower() == _PKL_EXT:
            npy_path = self.encoding_path.with_suffix(_NPY_EXT)
            logger.warning(
                "encoding_path '%s' uses the legacy .pkl suffix; "
                "storing encodings in '%s' instead",
                self.encoding_path, npy_path,
            )
            self.encoding_path = npy_path
        self._allow_legacy_pickle = bool(allow_legacy_pickle)
        self.persons = []          # [{"name": str, "image_path": str}, ...]
        self.encodings = []        # list of np.ndarray
        self._version = 0
        self._cached_names = []
        self._safe_dir = self.db_path.parent.resolve()
        self._lock = threading.Lock()
        self._load()

    @property
    def version(self):
        """Database version — used by recognizer cache invalidation."""
        return self._version

    def _validate_image_path(self, image_path: str) -> Path:
        """Validate that image_path resolves within the safe directory.

        Relative paths are resolved against the database's parent directory
        (safe_dir), not the current working directory.
        """
        path_obj = Path(image_path)
        if path_obj.is_absolute():
            resolved = path_obj.resolve()
        else:
            resolved = (self._safe_dir / path_obj).resolve()
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

        # Validate path *before* storing — prevents path-traversal in DB records.
        safe_path = self._validate_image_path(image_path)

        with self._lock:
            for person in self.persons:
                if person["name"] == name:
                    return False, f"Person '{name}' already exists"

            self.persons.append({"name": name, "image_path": str(safe_path)})
            self.encodings.append(encoding)
            self._cached_names.append(name)
            self._version += 1
            self._save()
            return True, f"Added: {name}"

    def remove_person(self, name: str, *, delete_image: bool = False):
        """Remove a single person.

        Args:
            name: Person name to remove.
            delete_image: Also delete the reference photo from disk.
                Defaults to False — removing a database record must never
                destroy the user's source image (it may be the only copy).
        """
        with self._lock:
            for i, person in enumerate(self.persons):
                if person["name"] == name:
                    if delete_image:
                        self._delete_image_file(person)
                    del self.persons[i]
                    del self.encodings[i]
                    del self._cached_names[i]
                    self._version += 1
                    self._save()
                    return True, f"Removed: {name}"
            return False, f"Not found: {name}"

    def _delete_image_file(self, person: dict) -> None:
        """Delete a person's image file, refusing paths outside the safe dir.

        Caller must hold self._lock.
        """
        try:
            img_path = self._validate_image_path(person["image_path"])
            if img_path.exists():
                img_path.unlink(missing_ok=True)
        except ValueError:
            logger.warning(
                "Skipping deletion of image for '%s': path outside safe directory",
                person["name"],
            )

    def remove_persons(self, names: list, *, delete_image: bool = False):
        """Remove multiple persons at once.

        Args:
            names: Person names to remove.
            delete_image: Also delete reference photos from disk (see
                remove_person). Defaults to False.
        """
        removed = []
        not_found = []
        to_remove_indices = set()

        with self._lock:
            for name in names:
                found = False
                for i, person in enumerate(self.persons):
                    if person["name"] == name:
                        if delete_image:
                            self._delete_image_file(person)
                        to_remove_indices.add(i)
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
            self._save()
            return removed, not_found

    def get_names(self) -> list:
        """Return all registered names."""
        with self._lock:
            return self._cached_names.copy()

    def get_encodings_and_names(self) -> tuple:
        """Return (encodings, names) as shallow copies.

        Copying the two lists (references only, the ndarrays are shared)
        keeps a concurrent add/remove from corrupting a recognizer cache
        rebuild that is already in progress.
        """
        with self._lock:
            return list(self.encodings), list(self._cached_names)

    def save(self):
        """Persist the database to disk (thread-safe, atomic write)."""
        with self._lock:
            self._save()

    def load(self):
        """Load the database from disk (thread-safe)."""
        with self._lock:
            self._load()

    def _save(self):
        """Internal save — caller must hold self._lock."""
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

        except (IOError, OSError) as e:
            raise DatabaseError(
                f"Failed to save database: {e}", db_path=str(self.db_path)
            ) from e
        except ValueError as e:
            raise SerializationError(
                f"Failed to serialize database: {e}", db_path=str(self.db_path)
            ) from e

    def _load(self):
        """Internal load — caller must hold self._lock."""
        try:
            if self.db_path.exists():
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.persons = data.get("persons", [])

            self.encodings = self._load_encodings()
            self._cached_names = [p["name"] for p in self.persons]
            self._version += 1
        except SerializationError:
            # Raised by _load_encodings for corrupt/wrong-shape .npy files.
            # Must not be re-wrapped as a plain DatabaseError by the IOError
            # handler below (SerializationError inherits from OSError).
            raise
        except json.JSONDecodeError as e:
            raise SerializationError(
                f"Failed to parse database JSON: {e}", db_path=str(self.db_path)
            ) from e
        except IOError as e:
            raise DatabaseError(
                f"Failed to load database: {e}", db_path=str(self.db_path)
            ) from e

    def _load_encodings(self) -> list:
        """Load encodings from .npy file, with optional legacy .pkl migration."""
        enc_path = self.encoding_path

        # Try the primary path first
        if enc_path.exists():
            return self._load_npy(enc_path)

        # If the primary path is .npy, also check for a legacy .pkl file.
        # Pickle deserialization can execute arbitrary code, so migration is
        # opt-in via allow_legacy_pickle.
        if enc_path.suffix == _NPY_EXT:
            legacy_path = enc_path.with_suffix(_PKL_EXT)
            if legacy_path.exists():
                if self._allow_legacy_pickle:
                    return self._migrate_pkl_to_npy(legacy_path, enc_path)
                logger.warning(
                    "Ignoring legacy pickle encodings file '%s' — pickle "
                    "deserialization can execute arbitrary code. Pass "
                    "allow_legacy_pickle=True to FaceDatabase to migrate it "
                    "to the safe .npy format.",
                    legacy_path,
                )

        return []

    def _load_npy(self, path: Path) -> list:
        """Load encodings from a .npy file."""
        try:
            stacked = np.load(str(path), allow_pickle=False)
        except Exception as e:
            raise SerializationError(
                f"Failed to load encodings file '{path}': {e}",
                db_path=str(path),
            ) from e
        if stacked.ndim == 2 and stacked.shape[0] > 0:
            if stacked.shape[1] != 512:
                raise SerializationError(
                    f"Encodings file '{path}' has unexpected shape "
                    f"{stacked.shape}, expected (N, 512)",
                    db_path=str(path),
                )
            return [stacked[i] for i in range(stacked.shape[0])]
        return []

    def _migrate_pkl_to_npy(self, pkl_path: Path, npy_path: Path) -> list:
        """Migrate a legacy .pkl encoding file to safe .npy format.

        Only called when allow_legacy_pickle=True — pickle.load can execute
        arbitrary code embedded in the file.
        """
        import pickle
        logger.warning(
            "Migrating legacy %s to %s — only do this with files you trust, "
            "pickle deserialization can execute arbitrary code",
            pkl_path.name, npy_path.name,
        )
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

    def clear(self, *, delete_images: bool = False):
        """Clear the database and delete persisted files.

        Args:
            delete_images: Also delete the referenced photos from disk.
                Defaults to False — clearing the registry must not destroy
                the user's source images.
        """
        with self._lock:
            if delete_images:
                for person in self.persons:
                    self._delete_image_file(person)

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
