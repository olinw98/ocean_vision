from fathomfollow.perception.tracker import SimpleTracker
from fathomfollow.perception.types import DetectionRecord


def test_stable_track_id() -> None:
    tr = SimpleTracker(max_gap=3)
    d1 = [DetectionRecord(0, 0, (0.5, 0.5, 0.2, 0.2), 0.9)]
    d2 = [DetectionRecord(1, 0, (0.51, 0.5, 0.2, 0.2), 0.85)]
    t1 = tr.update(d1)
    t2 = tr.update(d2)
    assert t1[0].track_id == t2[0].track_id


def test_track_drops_after_gap() -> None:
    tr = SimpleTracker(max_gap=2)
    tr.update([DetectionRecord(0, 0, (0.5, 0.5, 0.2, 0.2), 0.9)])
    tr.update([])
    tr.update([])
    tr.update([])
    assert tr.update([]) == []


def test_select_active_deterministic() -> None:
    tr = SimpleTracker()
    tracks = tr.update([DetectionRecord(0, 0, (0.5, 0.5, 0.2, 0.2), 0.9)])
    active = tr.select_active(tracks)
    assert active is not None
    assert active.track_id == 1
