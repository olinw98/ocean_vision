from __future__ import annotations

from dataclasses import replace

from fathomfollow.perception.types import DetectionRecord, Track


class SimpleTracker:
    """Minimal tracker for tests; stable IDs with max_gap support."""

    def __init__(self, max_gap: int = 5) -> None:
        self._max_gap = max_gap
        self._tracks: dict[int, Track] = {}
        self._next_id = 1
        self._frame = 0

    def update(self, dets: list[DetectionRecord]) -> list[Track]:
        self._frame += 1
        if not dets:
            to_drop = [
                tid
                for tid, tr in self._tracks.items()
                if self._frame - tr.last_seen_frame > self._max_gap
            ]
            for tid in to_drop:
                del self._tracks[tid]
            return list(self._tracks.values())

        det = max(dets, key=lambda d: d.confidence)
        if self._tracks:
            tid = min(self._tracks.keys())
            tr = self._tracks[tid]
            self._tracks[tid] = replace(
                tr,
                last_bbox=det.bbox,
                last_seen_frame=self._frame,
                history=(tr.history or []) + [det.bbox],
            )
        else:
            self._tracks[self._next_id] = Track(
                track_id=self._next_id,
                last_bbox=det.bbox,
                last_seen_frame=self._frame,
            )
            self._next_id += 1
        return list(self._tracks.values())

    def select_active(self, tracks: list[Track]) -> Track | None:
        if not tracks:
            return None
        return max(tracks, key=lambda t: t.last_seen_frame)
