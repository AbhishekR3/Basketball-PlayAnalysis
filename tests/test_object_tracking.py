"""P0-F (#12): confirmed tracks must carry their matched detection's confidence,
not a positionally misaligned score or 0.0.

Exercises the vendored DeepSORT Tracker directly -- Object_Tracking.py runs a full
YOLO+video pipeline at import, so the tracking mechanism is validated here at the
Track/Detection level where the #12 fix lives.
"""

import numpy as np

from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker


def _make_detection(box, confidence, class_id):
    # Distinct appearance feature per object so the matcher keeps them separate
    feature = np.full(128, float(confidence), dtype=np.float32)
    return Detection(box, confidence, feature, class_id)


def test_track_init_confidence_is_none():
    metric = nn_matching.NearestNeighborDistanceMetric("euclidean", 0.4, None)
    tracker = Tracker(metric, n_init=2)
    tracker.update([_make_detection([100, 100, 20, 40], 0.8, 'Basketball')])
    # Freshly initiated (not yet updated) track has no detection confidence yet
    assert tracker.tracks[0].confidence is None


def test_confirmed_tracks_carry_matched_confidence_no_swap():
    metric = nn_matching.NearestNeighborDistanceMetric("euclidean", 0.4, None)
    tracker = Tracker(metric, n_init=2)

    # Two well-separated detections with distinct, known confidences
    box_a, conf_a = [100, 100, 20, 40], 0.8
    box_b, conf_b = [400, 400, 20, 40], 0.9

    for _ in range(4):
        tracker.predict()
        tracker.update([
            _make_detection(box_a, conf_a, 'Basketball'),
            _make_detection(box_b, conf_b, 'Team_A'),
        ])

    confirmed = [t for t in tracker.tracks if t.is_confirmed()]
    assert len(confirmed) >= 2

    confidences = sorted(t.confidence for t in confirmed)
    # Exactly the two detector confidences, and never the bogus 0.0
    assert confidences == [conf_a, conf_b]
    assert all(t.confidence not in (0.0, None) for t in confirmed)

    # Not swapped: the track near box_a's center carries conf_a, box_b -> conf_b.
    # center_x = top_left_x + w/2 (xyah from the Kalman state mean)
    for t in confirmed:
        center_x = t.mean[0]
        if center_x < 250:
            assert abs(t.confidence - conf_a) < 1e-6
        else:
            assert abs(t.confidence - conf_b) < 1e-6
