"""Lightweight multi-object tracker for person detections.

V2 improvements over simple centroid matching:
- centroid + IoU association
- short-term constant-velocity prediction
- exponential bounding-box smoothing
- temporary occlusion handling
- stable track lifecycle
- trajectory history
- no SciPy dependency
"""

from collections import OrderedDict, deque
import math


class Track:
    def __init__(self, object_id, bbox, smoothing=0.55, trail_length=20):
        self.object_id = object_id
        self.bbox = tuple(map(int, bbox))
        self.raw_bbox = self.bbox
        self.centroid = self._centroid(self.bbox)
        self.previous_centroid = self.centroid
        self.velocity = (0.0, 0.0)

        self.smoothing = smoothing
        self.disappeared = 0
        self.hits = 1
        self.age = 1
        self.visible = True

        self.trail = deque(maxlen=max(1, trail_length))
        self.trail.append(self.centroid)

    @staticmethod
    def _centroid(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _lerp(old, new, alpha):
        return old + alpha * (new - old)

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection = iw * ih

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection

        return intersection / union if union > 0 else 0.0

    def predict(self):
        """Predict the next centroid using constant velocity."""
        vx, vy = self.velocity
        cx, cy = self.centroid
        return cx + vx, cy + vy

    def distance_to(self, bbox):
        px, py = self.predict()
        bx, by = self._centroid(bbox)
        return math.hypot(px - bx, py - by)

    def association_cost(self, bbox, max_distance):
        """Return a lower-is-better association cost.

        Combines normalized predicted-centroid distance and IoU.
        A small IoU bonus helps preserve IDs when people overlap/cross.
        """
        distance = self.distance_to(bbox)
        iou = self._iou(self.bbox, bbox)

        if distance > max_distance and iou < 0.05:
            return float("inf")

        distance_score = min(distance / max_distance, 2.0)
        iou_score = 1.0 - iou

        # Distance is the main signal; IoU stabilizes nearby/crossing tracks.
        return 0.72 * distance_score + 0.28 * iou_score

    def update(self, bbox):
        """Update a track from a real detection and smooth its box."""
        bbox = tuple(map(int, bbox))
        new_cx, new_cy = self._centroid(bbox)

        old_cx, old_cy = self.centroid
        instant_vx = new_cx - old_cx
        instant_vy = new_cy - old_cy

        # Smooth velocity separately so a single noisy frame does not
        # cause an exaggerated prediction on the next frame.
        self.velocity = (
            0.65 * self.velocity[0] + 0.35 * instant_vx,
            0.65 * self.velocity[1] + 0.35 * instant_vy,
        )

        alpha = self.smoothing
        x1, y1, x2, y2 = self.bbox
        nx1, ny1, nx2, ny2 = bbox

        smoothed = (
            round(self._lerp(x1, nx1, alpha)),
            round(self._lerp(y1, ny1, alpha)),
            round(self._lerp(x2, nx2, alpha)),
            round(self._lerp(y2, ny2, alpha)),
        )

        self.previous_centroid = self.centroid
        self.bbox = smoothed
        self.raw_bbox = bbox
        self.centroid = self._centroid(smoothed)
        self.disappeared = 0
        self.hits += 1
        self.age += 1
        self.visible = True
        self.trail.append((round(self.centroid[0]), round(self.centroid[1])))

    def mark_missing(self):
        """Predict a short-lived missing track instead of jumping its ID."""
        self.disappeared += 1
        self.age += 1
        self.visible = False

        predicted_x, predicted_y = self.predict()
        cx, cy = self.centroid

        dx = predicted_x - cx
        dy = predicted_y - cy

        self.previous_centroid = self.centroid
        self.centroid = (predicted_x, predicted_y)

        x1, y1, x2, y2 = self.bbox
        self.bbox = (
            round(x1 + dx),
            round(y1 + dy),
            round(x2 + dx),
            round(y2 + dy),
        )
        self.trail.append((round(predicted_x), round(predicted_y)))


class CentroidTracker:
    """Stable lightweight tracker using centroid/IoU association."""

    def __init__(
        self,
        maxDisappeared=12,
        maxDistance=110,
        smoothing=0.55,
        trail_length=20,
    ):
        if maxDisappeared < 1:
            raise ValueError("maxDisappeared must be >= 1")
        if maxDistance <= 0:
            raise ValueError("maxDistance must be > 0")
        if not 0.0 <= smoothing <= 1.0:
            raise ValueError("smoothing must be between 0 and 1")

        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.bboxes = OrderedDict()
        self.disappeared = OrderedDict()
        self.trails = OrderedDict()

        self.tracks = OrderedDict()
        self.maxDisappeared = int(maxDisappeared)
        self.maxDistance = float(maxDistance)
        self.smoothing = float(smoothing)
        self.trail_length = int(max(1, trail_length))
        self.visible_count = 0

    def register(self, centroid, inputRect):
        """Register a new track."""
        object_id = self.nextObjectID
        track = Track(
            object_id,
            inputRect,
            smoothing=self.smoothing,
            trail_length=self.trail_length,
        )
        self.tracks[object_id] = track
        self._sync_compatibility_views()
        self.nextObjectID += 1

    def deregister(self, objectID):
        """Remove a track completely."""
        self.tracks.pop(objectID, None)
        self.objects.pop(objectID, None)
        self.bboxes.pop(objectID, None)
        self.disappeared.pop(objectID, None)
        self.trails.pop(objectID, None)

    def reset(self):
        """Reset active tracks and ID numbering."""
        self.nextObjectID = 0
        self.tracks.clear()
        self.objects.clear()
        self.bboxes.clear()
        self.disappeared.clear()
        self.trails.clear()
        self.visible_count = 0

    def _sync_compatibility_views(self):
        """Keep the old public dictionaries usable by existing code."""
        self.objects = OrderedDict(
            (track_id, (int(track.centroid[0]), int(track.centroid[1])))
            for track_id, track in self.tracks.items()
        )
        self.bboxes = OrderedDict(
            (track_id, track.bbox) for track_id, track in self.tracks.items()
        )
        self.disappeared = OrderedDict(
            (track_id, track.disappeared) for track_id, track in self.tracks.items()
        )
        self.trails = OrderedDict(
            (track_id, list(track.trail)) for track_id, track in self.tracks.items()
        )

    def _match(self, track_ids, rects):
        """Greedy global association using the lowest available cost."""
        candidates = []

        for row, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            for col, bbox in enumerate(rects):
                cost = track.association_cost(bbox, self.maxDistance)
                if math.isfinite(cost):
                    candidates.append((cost, row, col))

        # Matching globally by lowest cost avoids the common row-order bias
        # of independently choosing the nearest detection per track.
        candidates.sort(key=lambda item: item[0])

        used_rows = set()
        used_cols = set()
        matches = []

        for cost, row, col in candidates:
            if row in used_rows or col in used_cols:
                continue

            used_rows.add(row)
            used_cols.add(col)
            matches.append((row, col))

        return matches, used_rows, used_cols

    def update(self, rects):
        """Update tracks from current-frame bounding boxes.

        Returns:
            OrderedDict mapping object IDs to smoothed bounding boxes.
            Missing tracks remain briefly using motion prediction.
        """
        rects = [] if rects is None else [
            tuple(map(int, rect)) for rect in rects
        ]

        # No active tracks: register every detection.
        if not self.tracks:
            for rect in rects:
                self.register(self._centroid(rect), rect)
            self.visible_count = len(rects)
            self._sync_compatibility_views()
            return self.bboxes

        track_ids = list(self.tracks.keys())

        if not rects:
            for track_id in track_ids:
                track = self.tracks[track_id]
                track.mark_missing()
                if track.disappeared > self.maxDisappeared:
                    self.deregister(track_id)

            self.visible_count = 0
            self._sync_compatibility_views()
            return self.bboxes

        matches, used_rows, used_cols = self._match(track_ids, rects)

        for row, col in matches:
            self.tracks[track_ids[row]].update(rects[col])

        # Existing tracks without a matching detection are temporarily
        # predicted forward rather than immediately receiving a new ID.
        for row, track_id in enumerate(track_ids):
            if row not in used_rows and track_id in self.tracks:
                track = self.tracks[track_id]
                track.mark_missing()
                if track.disappeared > self.maxDisappeared:
                    self.deregister(track_id)

        # Every unmatched detection becomes a new track.
        for col, rect in enumerate(rects):
            if col not in used_cols:
                self.register(self._centroid(rect), rect)

        self.visible_count = len(rects)
        self._sync_compatibility_views()
        return self.bboxes

    @staticmethod
    def _centroid(rect):
        x1, y1, x2, y2 = rect
        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        )
