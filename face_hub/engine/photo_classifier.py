"""
Photo classification by face — group photos by the people in them.

Two modes:
  1. Gallery mode: pass a FaceRecognizer with registered people — faces
     matching the gallery are filed under the person's name.
  2. Discovery mode (no recognizer / empty gallery): faces are clustered
     by embedding cosine similarity and each cluster becomes an anonymous
     group ("person_001", "person_002", ...).

Faces that do not match the gallery fall through to clustering as well,
so unknown visitors still get grouped together.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from face_hub.types import (
    UNKNOWN_SENTINEL,
    ExportResult,
    PhotoClassificationResult,
    PhotoFace,
    PhotoGroup,
)
from face_hub.detector_protocol import DetectorProtocol
from face_hub.engine.face_recognizer import FaceRecognizer
from face_hub.exceptions import FaceHubError

logger = logging.getLogger("face_hub.photo_classifier")

ImageInput = Union[str, Path, np.ndarray]
ProgressCallback = Callable[[int, int, str], None]  # (done, total, photo_id)

CLUSTER_LABEL_PREFIX = "person_"


class _EmbeddingClusterer:
    """
    Greedy centroid clustering over L2-normalized embeddings with a
    post-assignment merge + reassign pass to fix order-dependent splits.

    Phase 1 (online): Each new embedding joins the cluster whose centroid
    has the highest cosine similarity if that similarity >= threshold;
    otherwise it starts a new cluster.  Every embedding is stored for
    later reassignment.

    Phase 2 (merge_pass): After all assignments, centroids that are
    within threshold are merged via Union-Find, then every stored
    embedding is reassigned to the closest merged centroid.  This
    corrects both centroid drift and cross-contamination that the
    greedy pass cannot avoid.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        self._centroids: List[np.ndarray] = []   # L2-normalized means
        self._counts: List[int] = []             # members per cluster
        self._store: List[Tuple[np.ndarray, str, int]] = []  # (emb, photo_id, face_idx)

    @property
    def cluster_count(self) -> int:
        return len(self._centroids)

    def assign(
        self, embedding: np.ndarray, photo_id: str = "", face_idx: int = 0,
    ) -> Tuple[int, float]:
        """
        Assign an embedding to a cluster.

        The embedding is stored internally so that merge_pass can
        reassign it later.

        Returns:
            (cluster_index, similarity_to_centroid). For a brand-new
            cluster the similarity is 1.0 by convention.
        """
        emb = np.asarray(embedding, dtype=np.float32).ravel()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        self._store.append((emb, photo_id, face_idx))

        best_idx, best_sim = -1, -1.0
        for i, centroid in enumerate(self._centroids):
            sim = float(emb @ centroid)
            if sim > best_sim:
                best_idx, best_sim = i, sim

        if best_idx >= 0 and best_sim >= self.threshold:
            # Running mean of the cluster, renormalized to unit length
            n = self._counts[best_idx]
            mean = (self._centroids[best_idx] * n + emb) / (n + 1)
            m_norm = np.linalg.norm(mean)
            if m_norm > 0:
                mean = mean / m_norm
            self._centroids[best_idx] = mean.astype(np.float32)
            self._counts[best_idx] = n + 1
            return best_idx, best_sim

        self._centroids.append(emb)
        self._counts.append(1)
        return len(self._centroids) - 1, 1.0

    def merge_pass(self) -> Tuple[Dict[int, int], List[Tuple[str, int, int]]]:
        """
        Merge nearby centroids, then reassign every stored embedding.

        Returns:
            (old_to_new, reassignments) where
            - old_to_new: cluster-index remapping (for gallery-label safety)
            - reassignments: list of (photo_id, face_idx, new_cluster_idx)
              for every stored embedding
        """
        k = len(self._centroids)
        if k <= 1:
            mapping = {i: i for i in range(k)}
            reassignments = [
                (pid, fi, 0 if k == 1 else 0)
                for _, pid, fi in self._store
            ]
            return mapping, reassignments

        # ── 1. Merge close centroids via Union-Find ────────────────
        parent = list(range(k))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        merges: List[Tuple[float, int, int]] = []
        for i in range(k):
            for j in range(i + 1, k):
                sim = float(self._centroids[i] @ self._centroids[j])
                if sim >= self.threshold:
                    merges.append((sim, i, j))

        merges.sort(reverse=True)
        for _, i, j in merges:
            if find(i) != find(j):
                union(i, j)

        # Compute merged centroids
        comp_members: Dict[int, List[int]] = {}
        for i in range(k):
            root = find(i)
            comp_members.setdefault(root, []).append(i)

        old_to_new: Dict[int, int] = {}
        new_centroids: List[np.ndarray] = []
        new_counts: List[int] = []

        for new_idx, (root, members) in enumerate(comp_members.items()):
            total_count = sum(self._counts[m] for m in members)
            weighted = sum(
                self._centroids[m] * self._counts[m] for m in members
            ) / total_count
            w_norm = np.linalg.norm(weighted)
            if w_norm > 0:
                weighted = weighted / w_norm
            new_centroids.append(weighted.astype(np.float32))
            new_counts.append(total_count)
            for m in members:
                old_to_new[m] = new_idx

        self._centroids = new_centroids
        self._counts = new_counts

        # ── 2. Reassign every stored embedding to closest centroid ─
        reassignments: List[Tuple[str, int, int]] = []
        new_counts_arr = [0] * len(new_centroids)

        for emb, pid, fi in self._store:
            best_idx, best_sim = -1, -1.0
            for ci, centroid in enumerate(self._centroids):
                sim = float(emb @ centroid)
                if sim > best_sim:
                    best_idx, best_sim = ci, sim

            if best_idx >= 0 and best_sim >= self.threshold:
                reassignments.append((pid, fi, best_idx))
                new_counts_arr[best_idx] += 1
            else:
                # Keep original assignment as fallback
                reassignments.append((pid, fi, -1))
                new_counts_arr[0] += 1  # won't matter; -1 is handled by caller

        self._counts = new_counts_arr
        return old_to_new, reassignments


class PhotoClassifier:
    """
    Classify photos by the faces they contain.

    Usage:
        classifier = PhotoClassifier(detector)                      # discovery mode
        classifier = PhotoClassifier(detector, recognizer)          # gallery mode
        result = classifier.classify_photos(["a.jpg", "b.jpg"])
        for label, group in result.groups.items():
            print(label, group.photo_ids)
    """

    GROUP_LABEL = "group_photo"

    def __init__(
        self,
        detector: DetectorProtocol,
        recognizer: Optional[FaceRecognizer] = None,
        cluster_threshold: float = 0.45,
        skip_low_quality: bool = True,
        group_threshold: int = 0,
        blur_threshold: Optional[float] = None,
    ):
        """
        Args:
            detector: Any object satisfying DetectorProtocol
                      (FaceDetector or a custom implementation).
            recognizer: Optional FaceRecognizer whose encoding cache holds
                        the registered gallery (see update_cache()).
            cluster_threshold: Cosine-similarity threshold for clustering
                               faces that don't match the gallery.
                               0.40 strict, 0.45 recommended, 0.50 loose.
            skip_low_quality: Ignore detections whose quality_pass is False.
            group_threshold: When a photo contains >= this many faces, all
                             faces are labeled as "group_photo" instead of
                             being individually classified.
                             0 (default) disables this feature.
            blur_threshold: If given, overrides the detector's blur_threshold
                            (Laplacian variance). Higher = stricter:
                            50  default, tolerates soft skin / mild blur
                            100 strict, rejects soft-focus portraits
                            200 very strict, only razor-sharp faces pass.
                            Only effective when detector is a FaceDetector.
        """
        if not isinstance(cluster_threshold, (int, float)) or not (0 < cluster_threshold <= 1):
            raise ValueError(
                f"cluster_threshold must be in (0, 1], got {cluster_threshold}"
            )
        self.detector = detector
        self.recognizer = recognizer
        self.cluster_threshold = float(cluster_threshold)
        self.skip_low_quality = skip_low_quality
        self.group_threshold = int(group_threshold)
        # Override detector's blur_threshold when explicitly provided and
        # the detector supports it (FaceDetector instances have the attr).
        if blur_threshold is not None and hasattr(detector, "blur_threshold"):
            detector.blur_threshold = float(blur_threshold)
        logger.info(
            "PhotoClassifier initialized (mode=%s, cluster_threshold=%.2f, group_threshold=%d)",
            "gallery" if recognizer is not None else "discovery",
            self.cluster_threshold,
            self.group_threshold,
        )

    # ── Public API ───────────────────────────────────────────────

    def classify_photos(
        self,
        images: Sequence[ImageInput],
        photo_ids: Optional[Sequence[str]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> PhotoClassificationResult:
        """
        Classify a batch of photos by face.

        Args:
            images: Image file paths (str / Path) or BGR numpy arrays.
            photo_ids: Optional explicit id per image. Defaults to the path
                       string for files, or "image_0001" style ids for arrays.
            progress_callback: Optional fn(done, total, photo_id) called
                               after each photo.

        Returns:
            PhotoClassificationResult with one PhotoGroup per person,
            per-face details, and bookkeeping lists for photos without
            usable faces or that failed to decode.

        Raises:
            ValueError: If photo_ids is given but its length differs from
                        images, or contains duplicates.
            FaceHubError: If detection fails on a photo.
        """
        ids = self._resolve_photo_ids(images, photo_ids)
        result = PhotoClassificationResult(total_photos=len(images))
        clusterer = _EmbeddingClusterer(self.cluster_threshold)
        total = len(images)

        for idx, (image, photo_id) in enumerate(zip(images, ids), start=1):
            frame = self._load_frame(image)
            if frame is None:
                logger.warning("Unreadable photo: %s", photo_id)
                result.unreadable_photos.append(photo_id)
                self._report_progress(progress_callback, idx, total, photo_id)
                continue

            detections = self.detector.detect_with_embeddings(frame)
            usable = [
                d for d in detections
                if d.has_embedding and (d.quality_pass or not self.skip_low_quality)
            ]

            if not usable:
                result.no_face_photos.append(photo_id)
                self._report_progress(progress_callback, idx, total, photo_id)
                continue

            # Group photo: faces >= group_threshold → label all as "group_photo"
            if self.group_threshold > 0 and len(usable) >= self.group_threshold:
                for det in usable:
                    self._record(result, photo_id, det, self.GROUP_LABEL, 1.0)
                self._report_progress(progress_callback, idx, total, photo_id)
                continue

            for face_idx, det in enumerate(usable):
                label, similarity = self._label_face(
                    det.embedding, clusterer, photo_id, face_idx,
                )
                self._record(result, photo_id, det, label, similarity)

            self._report_progress(progress_callback, idx, total, photo_id)

        # ── Merge pass: fuse clusters + reassign embeddings ────────
        if clusterer.cluster_count > 1:
            _old_to_new, reassignments = clusterer.merge_pass()
            self._apply_reassignments(result, reassignments, clusterer)

        # ── Build groups from the final face list (single source of truth)
        self._build_groups(result)

        logger.info(
            "Classified %d photos → %d groups, %d no-face, %d unreadable",
            result.total_photos, len(result.groups),
            len(result.no_face_photos), len(result.unreadable_photos),
        )
        return result

    # ── Internals ────────────────────────────────────────────────

    def _label_face(
        self, embedding: np.ndarray, clusterer: _EmbeddingClusterer,
        photo_id: str = "", face_idx: int = 0,
    ) -> Tuple[str, float]:
        """Match against the gallery first, fall through to clustering."""
        if self.recognizer is not None:
            name, confidence = self.recognizer.recognize(embedding)
            if name != UNKNOWN_SENTINEL:
                return name, confidence
        cluster_idx, similarity = clusterer.assign(embedding, photo_id, face_idx)
        return f"{CLUSTER_LABEL_PREFIX}{cluster_idx + 1:03d}", similarity

    @staticmethod
    def _apply_reassignments(
        result: PhotoClassificationResult,
        reassignments: List[Tuple[str, int, int]],
        clusterer: _EmbeddingClusterer,
    ) -> None:
        """Rewrite cluster labels based on merge-pass reassignments.

        Gallery labels (non-cluster names) are left untouched — only
        faces whose label starts with CLUSTER_LABEL_PREFIX are updated.
        Does NOT rebuild result.groups — that is done by _build_groups.
        """
        lookup: Dict[Tuple[str, int], int] = {}
        for pid, fi, new_idx in reassignments:
            if new_idx >= 0:
                lookup[(pid, fi)] = new_idx

        # Walk faces in the same order they were recorded (one per usable
        # detection, cluster faces only).  Gallery faces keep their label.
        cluster_face_counter: Dict[str, int] = {}
        for face in result.faces:
            if not face.label.startswith(CLUSTER_LABEL_PREFIX):
                continue
            fi = cluster_face_counter.get(face.photo_id, 0)
            cluster_face_counter[face.photo_id] = fi + 1

            new_idx = lookup.get((face.photo_id, fi))
            if new_idx is not None:
                face.label = f"{CLUSTER_LABEL_PREFIX}{new_idx + 1:03d}"

    @staticmethod
    def _build_groups(result: PhotoClassificationResult) -> None:
        """Rebuild result.groups from the (final) result.faces list.

        This is the single source of truth for groups — called once at
        the end of classify_photos, after all label mutations are done.
        """
        groups: Dict[str, PhotoGroup] = {}
        for face in result.faces:
            group = groups.get(face.label)
            if group is None:
                group = PhotoGroup(label=face.label)
                groups[face.label] = group
            if face.photo_id not in group.photo_ids:
                group.photo_ids.append(face.photo_id)
            group.face_count += 1
        result.groups = groups

    @staticmethod
    def _record(
        result: PhotoClassificationResult,
        photo_id: str,
        detection,
        label: str,
        similarity: float,
    ) -> None:
        result.faces.append(PhotoFace(
            photo_id=photo_id,
            bbox=detection.bbox,
            det_confidence=detection.confidence,
            label=label,
            similarity=float(similarity),
        ))

    @staticmethod
    def _load_frame(image: ImageInput) -> Optional[np.ndarray]:
        """Return a BGR frame, or None if the image cannot be decoded."""
        if isinstance(image, np.ndarray):
            return image if image.size > 0 else None
        frame = cv2.imread(str(image))
        return frame if frame is not None and frame.size > 0 else None

    @staticmethod
    def _resolve_photo_ids(
        images: Sequence[ImageInput], photo_ids: Optional[Sequence[str]]
    ) -> List[str]:
        if photo_ids is not None:
            if len(photo_ids) != len(images):
                raise ValueError(
                    f"photo_ids ({len(photo_ids)}) and images ({len(images)}) "
                    "must have the same length"
                )
            ids = [str(p) for p in photo_ids]
            if len(set(ids)) != len(ids):
                raise ValueError("photo_ids must not contain duplicates")
            return ids

        ids: List[str] = []
        for i, image in enumerate(images, start=1):
            if isinstance(image, np.ndarray):
                ids.append(f"image_{i:04d}")
            else:
                ids.append(str(image))
        return ids

    @staticmethod
    def _report_progress(
        callback: Optional[ProgressCallback], done: int, total: int, photo_id: str
    ) -> None:
        if callback is not None:
            try:
                callback(done, total, photo_id)
            except Exception:  # noqa: BLE001 — progress hooks must never break the batch
                logger.exception("progress_callback failed for %s", photo_id)


def classify_photos(
    images: Sequence[ImageInput],
    detector: Optional[DetectorProtocol] = None,
    recognizer: Optional[FaceRecognizer] = None,
    cluster_threshold: float = 0.45,
    group_threshold: int = 0,
    skip_low_quality: bool = True,
    blur_threshold: Optional[float] = None,
    photo_ids: Optional[Sequence[str]] = None,
    progress_callback: Optional[ProgressCallback] = None,
    **detector_kwargs,
) -> PhotoClassificationResult:
    """
    One-shot helper: classify photos by face without managing components.

    Builds a default FaceDetector when none is given (requires insightface);
    extra keyword arguments are forwarded to its constructor.

    Usage:
        from face_hub import classify_photos
        result = classify_photos(["party1.jpg", "party2.jpg"], device="auto")
        print(result.summary())

    Quality control:
        # Strict mode — reject blurry / soft-focus faces:
        result = classify_photos(images, skip_low_quality=True, blur_threshold=100)

        # Lenient mode — keep all detected faces regardless of quality:
        result = classify_photos(images, skip_low_quality=False)
    """
    if detector is None:
        from face_hub.engine.face_detector import FaceDetector
        detector = FaceDetector(**detector_kwargs)
    elif detector_kwargs:
        logger.warning("detector_kwargs are ignored when a detector is provided")

    classifier = PhotoClassifier(
        detector,
        recognizer=recognizer,
        cluster_threshold=cluster_threshold,
        group_threshold=group_threshold,
        skip_low_quality=skip_low_quality,
        blur_threshold=blur_threshold,
    )
    return classifier.classify_photos(
        images, photo_ids=photo_ids, progress_callback=progress_callback
    )


# ── Folder export ──────────────────────────────────────────────

_INVALID_FOLDER_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _sanitize_folder_name(label: str) -> str:
    """Make a label safe to use as a folder name on all platforms."""
    name = _INVALID_FOLDER_CHARS.sub("_", label).strip().strip(".")
    return name or "_unnamed"


def _resolve_conflict(target: Path, on_conflict: str) -> Optional[Path]:
    """
    Resolve a filename conflict in the target folder.

    Returns the final path to write, or None to skip this file.
    """
    if not target.exists():
        return target
    if on_conflict == "overwrite":
        return target
    if on_conflict == "skip":
        return None
    # "rename" — append _1, _2, ... before the extension
    for i in range(1, 1000):
        candidate = target.with_name(f"{target.stem}_{i}{target.suffix}")
        if not candidate.exists():
            return candidate
    raise FaceHubError(f"Could not resolve filename conflict for {target}")


def _annotate_image(
    image_path: str,
    faces: List,
    target_label: str,
) -> Optional[np.ndarray]:
    """Draw bounding boxes and person labels on a copy of the image.

    Only draws faces whose label matches *target_label*.
    Returns the annotated BGR image, or None if the file cannot be read.
    """
    frame = cv2.imread(image_path)
    if frame is None or frame.size == 0:
        return None

    h, w = frame.shape[:2]
    # Scale line thickness and font size relative to image size
    thickness = max(1, int(min(h, w) / 300) + 1)
    font_scale = max(0.4, min(h, w) / 800)

    for face in faces:
        if face.photo_id != image_path or face.label != target_label:
            continue
        x1, y1, x2, y2 = face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2
        # Green box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        # Label background + text
        text = face.label
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness,
        )
        text_y = max(y1 - 4, th + 4)
        cv2.rectangle(
            frame,
            (x1, text_y - th - 4),
            (x1 + tw + 6, text_y + baseline),
            (0, 255, 0),
            cv2.FILLED,
        )
        cv2.putText(
            frame, text, (x1 + 3, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness,
        )
    return frame


def export_to_folders(
    result: PhotoClassificationResult,
    output_dir: Union[str, Path],
    mode: str = "copy",
    include_no_face: bool = True,
    no_face_label: str = "_no_face",
    on_conflict: str = "rename",
    annotate_faces: bool = False,
) -> ExportResult:
    """
    Export a classification result into per-person folders.

    One folder per group label is created under output_dir. A photo
    containing several people is exported into **every** folder it
    belongs to. Only photo ids that point to existing files are
    exported; others (e.g. array inputs) are reported as skipped.

    Args:
        result: PhotoClassificationResult from classify_photos().
        output_dir: Root folder to create person folders in.
        mode: "copy" (keep originals) or "move".
        include_no_face: Also export no_face_photos into a folder
                         named no_face_label.
        no_face_label: Folder name for photos without a usable face.
        on_conflict: What to do when a file with the same name already
                     exists in a target folder —
                     "rename" (default, append _1, _2, ...),
                     "skip", or "overwrite".

    Returns:
        ExportResult with exported paths per label, skipped photo ids,
        and per-photo error messages.

    Raises:
        ValueError: On invalid mode or on_conflict values.
        FaceHubError: If output_dir cannot be created.
    """
    if mode not in ("copy", "move"):
        raise ValueError(f"mode must be 'copy' or 'move', got {mode!r}")
    if on_conflict not in ("rename", "skip", "overwrite"):
        raise ValueError(
            f"on_conflict must be 'rename', 'skip' or 'overwrite', "
            f"got {on_conflict!r}"
        )

    root = Path(output_dir)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise FaceHubError(f"Cannot create output_dir {root}: {e}") from e

    export = ExportResult()
    transfer = shutil.copy2 if mode == "copy" else shutil.move
    moved_sources: Dict[str, str] = {}  # original photo_id → first exported path
    skipped_seen = set()                # O(1) membership for export.skipped

    jobs: List[Tuple[str, str]] = []  # (label, photo_id)
    for label, group in result.groups.items():
        jobs.extend((label, photo_id) for photo_id in group.photo_ids)
    if include_no_face:
        jobs.extend((no_face_label, photo_id) for photo_id in result.no_face_photos)

    for label, photo_id in jobs:
        src = Path(photo_id)
        # In move mode a multi-person photo is moved once, then copied from
        # its first destination into the remaining folders.
        already_moved = photo_id in moved_sources
        if not src.is_file() and not already_moved:
            if photo_id not in skipped_seen:
                skipped_seen.add(photo_id)
                export.skipped.append(photo_id)
            continue

        folder = root / _sanitize_folder_name(label)
        try:
            folder.mkdir(parents=True, exist_ok=True)
            target = _resolve_conflict(folder / src.name, on_conflict)
            if target is None:
                continue

            if annotate_faces and src.is_file():
                # Annotate: draw boxes on a copy of the image and write directly
                annotated = _annotate_image(photo_id, result.faces, label)
                if annotated is not None:
                    cv2.imwrite(str(target), annotated)
                    written = str(target)
                else:
                    # Fall back to plain copy if annotation fails
                    written = shutil.copy2(str(src), str(target))
            elif already_moved:
                written = shutil.copy2(moved_sources[photo_id], str(target))
            else:
                written = transfer(str(src), str(target))
                if mode == "move":
                    moved_sources[photo_id] = str(written)

            export.exported.setdefault(label, []).append(str(written))
        except (OSError, shutil.Error) as e:
            export.errors[photo_id] = str(e)
            logger.warning("Export failed for %s → %s: %s", photo_id, label, e)

    logger.info(
        "Exported %d file(s) into %d folder(s) under %s (%d skipped, %d errors)",
        export.total_files, len(export.exported), root,
        len(export.skipped), len(export.errors),
    )
    return export
