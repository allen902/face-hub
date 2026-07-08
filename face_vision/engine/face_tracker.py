"""
Face temporal tracking module.
Lightweight IoU-based multi-face tracking + identity voting smoothing.

How it works:
1. Detections in each frame are matched to existing tracks via IoU.
2. Each track keeps the last N recognition results.
3. Identity is confirmed when a majority vote is consistent across smooth_frames.
4. This suppresses single-frame misrecognition flicker.
"""

import numpy as np
import logging

from face_vision.types import UNKNOWN_SENTINEL, TrackedFace, BBox

logger = logging.getLogger("face_vision.tracker")


def _iou(boxA, boxB):
    """Compute Intersection over Union of two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter_area / float(areaA + areaB - inter_area)


class FaceTrack:
    """Tracking state for a single face."""

    __slots__ = ('id', 'bbox', 'name_history', 'conf_history',
                 'quality_history', 'frames_since_update', 'total_frames',
                 'confirmed_name', 'confirmed_conf', 'latest_name', 'latest_conf')

    def __init__(self, track_id, bbox):
        self.id = track_id
        self.bbox = bbox                  # current bbox as (x1, y1, x2, y2)
        self.name_history = []            # recent recognition names (up to smooth_frames*3)
        self.conf_history = []            # corresponding confidences
        self.quality_history = []         # corresponding quality flags
        self.frames_since_update = 0      # consecutive unmatched frames
        self.total_frames = 1             # total tracked frames
        self.confirmed_name = UNKNOWN_SENTINEL
        self.confirmed_conf = 0.0
        self.latest_name = UNKNOWN_SENTINEL
        self.latest_conf = 0.0

    def update(self, bbox, name, conf, quality_pass=True):
        """Update the track with a new detection."""
        self.bbox = bbox
        self.frames_since_update = 0
        self.total_frames += 1
        self.name_history.append(name)
        self.conf_history.append(conf)
        self.quality_history.append(quality_pass)
        self.latest_name = name
        self.latest_conf = conf

    def mark_missed(self):
        """Mark the track as missed in the current frame."""
        self.frames_since_update += 1

    def is_stale(self, max_missed=10):
        """Return True if the track has expired."""
        return self.frames_since_update >= max_missed

    def _majority_vote(self, smooth_frames):
        """
        Majority vote over the last `smooth_frames` frames.
        Uses an adaptive threshold: lower requirement when few frames exist.

        Returns:
            (name, avg_confidence, is_confirmed)
        """
        if not self.name_history:
            return self.latest_name, self.latest_conf, False

        # Prefer quality-passed frames
        recent_qualified = [
            (n, c) for n, c, q in zip(
                self.name_history[-smooth_frames:],
                self.conf_history[-smooth_frames:],
                self.quality_history[-smooth_frames:]
            )
            if q
        ]

        # Fall back to all recent frames if none passed quality
        recent = recent_qualified if recent_qualified else list(zip(
            self.name_history[-smooth_frames:],
            self.conf_history[-smooth_frames:]
        ))

        if not recent:
            return self.latest_name, self.latest_conf, False

        votes = {}
        conf_sums = {}
        for name, conf in recent:
            votes[name] = votes.get(name, 0) + 1
            conf_sums[name] = conf_sums.get(name, 0.0) + conf

        best_name = max(votes, key=votes.get)
        best_votes = votes[best_name]
        avg_conf = conf_sums[best_name] / best_votes if best_votes > 0 else 0.0

        total = len(recent)
        if total >= 6:
            min_votes = total // 2 + 1          # >50%
        elif total >= 3:
            min_votes = total // 2 + 1          # >50%, 3 frames need 2 votes
        else:
            min_votes = total                   # 1-2 frames: 1 vote is enough

        is_confirmed = (best_votes >= min_votes
                        and best_name != UNKNOWN_SENTINEL
                        and avg_conf >= 0.30)

        return best_name, avg_conf, is_confirmed

    def resolve_identity(self, smooth_frames=5):
        """
        Resolve the final identity for the current frame.

        Returns:
            (display_name, confidence, is_confirmed)
        """
        name, conf, is_confirmed = self._majority_vote(smooth_frames)
        if is_confirmed:
            self.confirmed_name = name
            self.confirmed_conf = conf
            return name, conf, True
        else:
            if self.latest_name != UNKNOWN_SENTINEL:
                return self.latest_name, self.latest_conf, False
            return self.latest_name, 0.0, False


class FaceTracker:
    """
    Multi-face tracker.
    - IoU matching between detections and tracks
    - Identity smoothing via majority vote
    - Automatic track creation / destruction
    """

    def __init__(self, smooth_frames=5, iou_threshold=0.30, max_missed=10):
        self.smooth_frames = smooth_frames
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks = []
        self._next_id = 0

    def update(self, detections, recognizer=None, known_encodings=None, known_names=None):
        """
        Update tracks with the current frame detections.

        Args:
            detections: list of DetectionWithEmbedding
            recognizer: FaceRecognizer instance (optional, used for recognition)
            known_encodings, known_names: deprecated / optional explicit gallery

        Returns:
            list of TrackedFace
        """
        # 1. Recognize each detection
        recognized = []
        for det in detections:
            bbox = det.bbox.to_tuple()
            det_conf = det.confidence
            embedding = det.embedding
            quality_pass = det.quality_pass

            name, rec_conf = UNKNOWN_SENTINEL, 0.0
            if embedding is not None and recognizer is not None:
                try:
                    name, rec_conf = recognizer.recognize(embedding, None, None)
                except Exception:
                    pass

            recognized.append({
                'bbox': bbox,
                'name': name,
                'conf': rec_conf,
                'det_conf': det_conf,
                'embedding': embedding,
                'quality_pass': quality_pass,
            })

        # 2. IoU match detections to existing tracks
        matched_track_ids = set()
        matched_det_ids = set()

        if self.tracks and recognized:
            iou_matrix = np.zeros((len(self.tracks), len(recognized)))
            for ti, track in enumerate(self.tracks):
                for di, det in enumerate(recognized):
                    iou_matrix[ti, di] = _iou(track.bbox, det['bbox'])

            while True:
                if iou_matrix.size == 0:
                    break
                max_iou = np.max(iou_matrix)
                if max_iou < self.iou_threshold:
                    break
                ti, di = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                det = recognized[di]
                self.tracks[ti].update(
                    det['bbox'], det['name'], det['conf'], det['quality_pass']
                )
                matched_track_ids.add(ti)
                matched_det_ids.add(di)
                iou_matrix[ti, :] = 0
                iou_matrix[:, di] = 0

        # 3. Mark unmatched tracks as missed
        for ti, track in enumerate(self.tracks):
            if ti not in matched_track_ids:
                track.mark_missed()

        # 4. Create new tracks for unmatched detections
        for di, det in enumerate(recognized):
            if di not in matched_det_ids:
                new_track = FaceTrack(self._next_id, det['bbox'])
                self._next_id += 1
                new_track.update(det['bbox'], det['name'], det['conf'], det['quality_pass'])
                self.tracks.append(new_track)

        # 5. Remove stale tracks
        self.tracks = [t for t in self.tracks if not t.is_stale(self.max_missed)]

        # 6. Limit history length
        max_history = self.smooth_frames * 3
        for t in self.tracks:
            if len(t.name_history) > max_history:
                t.name_history = t.name_history[-self.smooth_frames:]
                t.conf_history = t.conf_history[-self.smooth_frames:]
                t.quality_history = t.quality_history[-self.smooth_frames:]

        # 7. Resolve identities and build output
        results = []
        for track in self.tracks:
            display_name, display_conf, is_confirmed = track.resolve_identity(
                self.smooth_frames
            )
            x1, y1, x2, y2 = track.bbox
            results.append(TrackedFace(
                track_id=track.id,
                bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                name=display_name,
                confidence=display_conf,
                det_confidence=0.0,
                is_confirmed=is_confirmed,
                quality_pass=True,
            ))

        return results

    def reset(self):
        """Reset all tracks."""
        self.tracks.clear()
        self._next_id = 0

    @property
    def track_count(self):
        """Number of currently active tracks."""
        return len(self.tracks)
