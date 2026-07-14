"""
Face recognition module.
1:N cosine-similarity matching with an internal encoding matrix cache.
Embeddings are provided by FaceDetector.detect_with_embeddings() to avoid duplicate inference.
"""

import numpy as np
import logging

from face_hub.types import UNKNOWN_SENTINEL

logger = logging.getLogger("face_hub.recognizer")


class FaceRecognizer:
    """Face recognizer — 1:N feature matching + encoding matrix cache."""

    def __init__(self, tolerance=0.45, device=None):
        """
        Args:
            tolerance: Cosine-similarity threshold.
                       For buffalo_l 512-dim normed embeddings:
                       0.40 strict, 0.45 recommended, 0.50 loose, >0.50 too loose.
            device: Deprecated / reserved for future use.
        """
        self.tolerance = tolerance
        # Encoding cache — rebuilt only when the database changes
        self._cached_encodings = None   # np.ndarray (N, 512) or None
        self._cached_names = []         # list of str
        self._db_version = -1           # compared against database version
        logger.info("FaceRecognizer initialized (tolerance=%s)", tolerance)

    @property
    def cached_names(self):
        """Names currently stored in the encoding cache."""
        return self._cached_names

    def update_cache(self, known_encodings, known_names, db_version=0):
        """
        Update the encoding matrix cache (call only when the database changes).

        Args:
            known_encodings: list of np.ndarray
            known_names:     list of str
            db_version:      database version number (cache is skipped if unchanged)

        Returns:
            bool — True if the cache was actually rebuilt.

        Raises:
            ValueError: If encodings and names have different lengths.
        """
        if db_version == self._db_version and self._cached_encodings is not None:
            return False  # cache is still valid

        if len(known_encodings) != len(known_names):
            raise ValueError(
                f"known_encodings ({len(known_encodings)}) and "
                f"known_names ({len(known_names)}) must have the same length"
            )

        if len(known_encodings) == 0:
            self._cached_encodings = None
            self._cached_names = []
        else:
            mat = np.array(known_encodings, dtype=np.float32)
            # L2-normalise each row so dot product == cosine similarity
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            np.divide(mat, np.maximum(norms, 1e-12, out=norms), out=mat)
            self._cached_encodings = mat
            self._cached_names = list(known_names)

        self._db_version = db_version
        return True

    def recognize(self, unknown_encoding, known_encodings=None, known_names=None):
        """
        1:N recognition: cosine-similarity comparison against the registered gallery.

        Backwards-compatible with explicit (known_encodings, known_names),
        but update_cache() is recommended for speed.

        Args:
            unknown_encoding: np.ndarray (512,) — query face embedding
            known_encodings:  list of np.ndarray — optional explicit gallery
            known_names:      list of str — optional explicit names

        Returns:
            (name, confidence) where confidence is in [0.0, 1.0].
        """
        # Decide between explicit arguments and the internal cache
        if known_encodings is not None and len(known_encodings) > 0:
            encodings = np.array(known_encodings, dtype=np.float32)
            names = known_names if known_names else []
            if len(names) != len(encodings):
                logger.warning(
                    "known_encodings (%d) and known_names (%d) length mismatch",
                    len(encodings), len(names),
                )
                return UNKNOWN_SENTINEL, 0.0
            # L2-normalise gallery rows
            norms = np.linalg.norm(encodings, axis=1, keepdims=True)
            np.divide(encodings, np.maximum(norms, 1e-12, out=norms), out=encodings)
        elif self._cached_encodings is not None and len(self._cached_names) > 0:
            encodings = self._cached_encodings  # already normalised by update_cache
            names = self._cached_names
        else:
            return UNKNOWN_SENTINEL, 0.0

        if unknown_encoding is None or len(names) == 0:
            return UNKNOWN_SENTINEL, 0.0

        # Validate encoding dimension
        if hasattr(encodings, 'shape') and encodings.ndim == 2:
            expected_dim = encodings.shape[1]
        else:
            expected_dim = 512
        if hasattr(unknown_encoding, 'shape') and unknown_encoding.size != expected_dim:
            logger.warning(
                "Encoding dimension mismatch: query has %d dims, gallery has %d",
                unknown_encoding.size, expected_dim,
            )
            return UNKNOWN_SENTINEL, 0.0

        # Ensure float32 without unnecessary copies
        if not isinstance(unknown_encoding, np.ndarray):
            unknown_encoding = np.asarray(unknown_encoding, dtype=np.float32)
        elif unknown_encoding.dtype != np.float32:
            unknown_encoding = unknown_encoding.astype(np.float32)
        unknown_encoding = unknown_encoding.ravel()

        norm = np.linalg.norm(unknown_encoding)
        if norm > 0:
            unknown_encoding = unknown_encoding / norm

        # Cosine similarity (dot product — both query and gallery are L2-normalised)
        similarities = unknown_encoding @ encodings.T

        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim < self.tolerance:
            return UNKNOWN_SENTINEL, 0.0

        return names[best_idx], best_sim
