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

    def _resolve_gallery(self, known_encodings, known_names):
        """
        Pick the gallery to match against: explicit arguments first,
        otherwise the internal cache.

        Returns:
            (encodings, names) — L2-normalised (N, D) float32 matrix and
            name list — or (None, []) when no usable gallery exists.
        """
        if known_encodings is not None and len(known_encodings) > 0:
            encodings = np.array(known_encodings, dtype=np.float32)
            names = known_names if known_names else []
            if len(names) != len(encodings):
                logger.warning(
                    "known_encodings (%d) and known_names (%d) length mismatch",
                    len(encodings), len(names),
                )
                return None, []
            # L2-normalise gallery rows
            norms = np.linalg.norm(encodings, axis=1, keepdims=True)
            np.divide(encodings, np.maximum(norms, 1e-12, out=norms), out=encodings)
            return encodings, names
        if self._cached_encodings is not None and len(self._cached_names) > 0:
            return self._cached_encodings, self._cached_names  # already normalised
        return None, []

    def recognize(self, unknown_encoding, known_encodings=None, known_names=None):
        """
        1:N recognition: cosine-similarity comparison against the registered gallery.

        Backwards-compatible with explicit (known_encodings, known_names),
        but update_cache() is recommended for speed.
        Note: passing an explicit but *empty* known_encodings falls back to
        the internal cache.

        Args:
            unknown_encoding: np.ndarray (512,) — query face embedding
            known_encodings:  list of np.ndarray — optional explicit gallery
            known_names:      list of str — optional explicit names

        Returns:
            (name, confidence) where confidence is in [0.0, 1.0].
        """
        results = self.recognize_batch(
            [unknown_encoding],
            known_encodings=known_encodings,
            known_names=known_names,
        )
        return results[0]

    def recognize_batch(self, unknown_encodings, known_encodings=None, known_names=None):
        """
        Batched 1:N recognition — one matrix multiply for the whole frame
        instead of one numpy call chain per face.

        Args:
            unknown_encodings: sequence of np.ndarray (512,) — one per face;
                               None entries are skipped and returned as unknown.
            known_encodings:   list of np.ndarray — optional explicit gallery
            known_names:       list of str — optional explicit names

        Returns:
            list of (name, confidence), same length as unknown_encodings.
        """
        results = [(UNKNOWN_SENTINEL, 0.0)] * len(unknown_encodings)

        encodings, names = self._resolve_gallery(known_encodings, known_names)
        if encodings is None or len(names) == 0:
            return results

        valid = [(i, e) for i, e in enumerate(unknown_encodings) if e is not None]
        if not valid:
            return results
        idxs = [i for i, _ in valid]
        mat = np.stack(
            [np.asarray(e, dtype=np.float32).ravel() for _, e in valid]
        ).astype(np.float32, copy=False)

        # Validate encoding dimension
        expected_dim = encodings.shape[1] if encodings.ndim == 2 else 512
        if mat.shape[1] != expected_dim:
            logger.warning(
                "Encoding dimension mismatch: queries have %d dims, gallery has %d",
                mat.shape[1], expected_dim,
            )
            return results

        # L2-normalise queries (zero vectors stay zero → similarity 0 → unknown)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        np.divide(mat, np.maximum(norms, 1e-12, out=norms), out=mat)

        # Cosine similarity for every query in one go
        sims = mat @ encodings.T
        best_idx = np.argmax(sims, axis=1)
        best_sim = sims[np.arange(len(idxs)), best_idx]

        for row, i in enumerate(idxs):
            if best_sim[row] >= self.tolerance:
                results[i] = (names[int(best_idx[row])], float(best_sim[row]))
        return results
